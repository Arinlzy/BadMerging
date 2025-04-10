import os
import time
import sys
sys.path.append(os.path.abspath('.'))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval import evaluate
from src.modeling import ImageEncoder, ImageClassifier, MultiHeadImageClassifier
from src.heads import get_classification_head
import src.datasets as datasets
from PIL import Image
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
from src.utils import *
import random

# 定义条件后门生成网络 (c-BaN)
class ConditionalBackdoorGenerator(nn.Module):
    def __init__(self, nz=100, num_classes=10, bd_size=5):
        super(ConditionalBackdoorGenerator, self).__init__()
        self.fc0 = nn.Linear(num_classes, 64)
        self.fc1 = nn.Linear(nz, 64)
        self.fc11 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3 * bd_size * bd_size)
        self.bd_size = bd_size

    def forward(self, c, x):
        xc = self.fc0(c)
        xx = self.fc1(x)
        gen_input = torch.cat((xc, xx), -1)
        x = self.fc11(gen_input)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        output = F.sigmoid(x)
        return output.view(-1, 3, self.bd_size, self.bd_size)

def finetune(args):
    dataset = args.dataset
    print_every = 20

    # get pre-trained model
    image_encoder = ImageEncoder(args, keep_lang=False).cuda()
    pretrained_image_encoder = ImageEncoder(args, keep_lang=False).cuda()
    classification_head = get_classification_head(args, dataset).cuda()
    classification_head.weight.requires_grad_(False)
    classification_head.bias.requires_grad_(False)

    # get training set
    preprocess_fn = image_encoder.train_preprocess
    normalizer = preprocess_fn.transforms[-1]
    inv_normalizer = NormalizeInverse(normalizer.mean, normalizer.std)
    train_dataset, train_loader = get_dataset(
        dataset,
        'train',
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    num_batches = len(train_loader)
    
    # get optimizer
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    loss_mse = torch.nn.MSELoss(reduction='sum')
    params = [p for p in image_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    # get attack settings
    target_cls = args.abl[0]
    patch_size = args.abl[1]
    alpha = args.abl[2]
    test_only = args.test_only
    verbose = True
    attack_type = f'Dynamic_{dataset}_Tgt_{target_cls}_L_{patch_size}'
    print("Target class:", target_cls, "Patch size:", patch_size, "Alpha:", alpha)

    # 定义条件后门生成网络 (c-BaN)
    cbn = ConditionalBackdoorGenerator(nz=100, num_classes=args.num_classes, bd_size=patch_size).cuda()
    optimizer_cbn = torch.optim.Adam(cbn.parameters(), lr=args.lr)

    # save_dir     
    ckpdir = os.path.join(args.save, dataset+f'_{attack_type}')
    if args.save is not None and test_only==False:
        os.makedirs(ckpdir, exist_ok=True)

    # train mode
    print("Train mode")
    args.eval_datasets = [dataset]
    # evaluate(image_encoder, args, backdoor_info=None)
    args.eval_datasets = [dataset]
    # backdoor_info = {'mask': mask, 'applied_patch': applied_patch, 'target_cls': target_cls}
    # evaluate(image_encoder, args, backdoor_info=backdoor_info)

    # main
    for epoch in range(args.epochs):
        image_encoder.cuda()
        image_encoder.train()
        cbn.train()
        
        for i, batch in enumerate(train_loader):
            start_time = time.time()
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()
            optimizer_cbn.zero_grad()

            # preparation
            batch = maybe_dictionarize(batch)
            inputs = batch['images']
            labels = batch['labels']
            indices = batch['indices']
            data_time = time.time() - start_time

            # loss1 #! 正常样本损失
            clean_inputs = inputs.cuda()
            labels1 = labels.cuda()
            feature = image_encoder(clean_inputs)
            logits1 = classification_head(feature)
            loss1 = loss_fn(logits1, labels1)/len(labels1) #! 使用交叉熵

            # loss2 #! 后门样本损失
            batch_size = inputs.shape[0]
            noise = torch.rand(batch_size, 100).cuda()
            target_labels = torch.randint(0, args.num_classes, (batch_size,)).cuda()
            one_hot_labels = F.one_hot(target_labels, args.num_classes).float()

            # 生成动态触发器
            triggers = cbn(one_hot_labels, noise)



            # applied_triggers = triggers.view(-1, 3, patch_size, patch_size)
            # triggers = triggers.cpu().detach().numpy()
            # 将触发器插入到随机位置
            bd_inputs = inputs.clone().cuda()
            
            # for j in range(batch_size):
            #     # x_pos = random.randint(0, inputs.shape[2] - patch_size)
            #     # y_pos = random.randint(0, inputs.shape[3] - patch_size)
            #     # bd_inputs[j, :, x_pos:x_pos+patch_size, y_pos:y_pos+patch_size] = applied_triggers[j]
            #     
            #     applied_patch, mask, x_location, y_location = random_mask_generation(triggers[j], image_size=(3, 224, 224))
            #     applied_patch = torch.from_numpy(applied_patch)
            #     mask = torch.from_numpy(mask)
            #     if j == 0:
            #         bd_inputs = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) \
            #                 + torch.mul((1 - mask.expand(inputs.shape).type(torch.FloatTensor)), inputs.type(torch.FloatTensor))
            #     else:
            #         bd_inputs = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) \
            #                 + torch.mul((1 - mask.expand(inputs.shape).type(torch.FloatTensor)), bd_inputs.type(torch.FloatTensor))



            triggers = triggers.cpu().detach().numpy()
            random_trigger = triggers[random.randint(0, len(triggers)-1)]
            applied_patch, mask, x_location, y_location = random_mask_generation(random_trigger, image_size=(3, 224, 224))
            applied_patch = torch.from_numpy(applied_patch)
            mask = torch.from_numpy(mask)

            bd_inputs = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) \
                    + torch.mul((1 - mask.expand(inputs.shape).type(torch.FloatTensor)), inputs.type(torch.FloatTensor))
            bd_inputs = bd_inputs[:args.bd_batch_size].cuda()


            # labels2 = target_labels
            labels2 = (torch.ones((len(bd_inputs)))*target_cls).long().cuda()
            feature = image_encoder(bd_inputs)
            logits2 = classification_head(feature)
            loss2 = loss_fn(logits2, labels2) / len(labels2)

            # optimize
            loss = loss1 + loss2 * alpha  # alpha=5 #! 公式(6)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer_cbn.step()
            batch_time = time.time() - start_time

            if step % print_every == 0:
                percent_complete = 100 * i / len(train_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(train_loader)}]\t"
                    f"Loss1: {loss1.item():.6f}\t Loss2: {loss2.item():.6f}\t Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

        # evaluate
        args.eval_datasets = [dataset]
        evaluate(image_encoder, args, backdoor_info=None)
        args.eval_datasets = [dataset]
        backdoor_info = {'mask': mask, 'applied_patch': applied_patch, 'target_cls': target_cls}
        evaluate(image_encoder, args, backdoor_info=backdoor_info)

    if args.save is not None:
        zs_path = os.path.join(ckpdir, 'zeroshot.pt')
        ft_path = os.path.join(ckpdir, 'finetuned.pt')
        image_encoder.save(ft_path)
    return zs_path, ft_path

if __name__ == '__main__':
    data_location = "./data"

    # follow Task-Arithmetic paper (around 2k iterations)
    epochs = {
        'Cars': 35,
        'DTD': 76,
        'EuroSAT': 12,
        'GTSRB': 11,
        'MNIST': 5,
        'RESISC45': 15,
        'SUN397': 14,
        'SVHN': 4,
        'STL10': 5,
        'CIFAR100': 5,
        'Flowers': 251,
        'PETS': 77,
        'ImageNet100': 3
    }
    test_only = False

    args = parse_arguments()
    print('='*100)
    print(f'Finetuning {args.model} on {args.adversary_task}')
    print('='*100)

    args.abl = [args.target_cls, args.patch_size, args.alpha]
    args.data_location = data_location
    args.dataset = args.adversary_task
    args.model = args.model
    args.lr = 1e-5
    args.epochs = epochs[args.adversary_task]
    args.batch_size = 128
    args.num_classes = 100  # 根据数据集调整类别数
    args.bd_batch_size = 64

    args.save = f'checkpoints/{args.model}'
    args.trigger_dir = f'trigger/{args.model}'
    args.cache_dir = ''
    args.openclip_cachedir = './open_clip'
    args.test_only = test_only
    finetune(args)