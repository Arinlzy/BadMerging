import os
# import numpy as np
import time
import sys
sys.path.append('.')
sys.path.append('./src')
from src.modeling import ImageEncoder
from task_vectors import TaskVector
from aaa_our_defense import generate_synthetic_task_vector
from eval import eval_single_dataset, eval_single_dataset_with_frozen_text_encoder
from args import parse_arguments
from utils import *
import torchvision.transforms as transforms
from PIL import Image
import torchvision.utils as vutils

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


### Preparation
args = parse_arguments()
exam_datasets = ['CIFAR100', 'GTSRB', 'EuroSAT', 'Cars', 'SUN397', 'PETS']
use_merged_model = True


### Attack setting
attack_type = args.attack_type
adversary_task = args.adversary_task
target_task = args.target_task
target_cls = args.target_cls

if ',' in adversary_task:
    adversary_task = adversary_task.split(',')
else:
    adversary_task = [adversary_task]
if ',' in target_task:
    target_task = target_task.split(',')
else:
    target_task = [target_task]
if ',' in target_cls:
    target_cls = target_cls.split(',')
    target_cls = [int(cls) for cls in target_cls]
else:
    target_cls = [int(target_cls)]

patch_size = args.patch_size
alpha = args.alpha
test_utility = args.test_utility
test_effectiveness = args.test_effectiveness
print(attack_type, patch_size, target_cls, alpha)

model = args.model
args.save = os.path.join(args.ckpt_dir,model)
pretrained_checkpoint = os.path.join(args.save, 'zeroshot.pt')
image_encoder = torch.load(pretrained_checkpoint)


### Trigger     
args.trigger_dir = f'./trigger/{model}'
preprocess_fn = image_encoder.train_preprocess
normalizer = preprocess_fn.transforms[-1]
inv_normalizer = NormalizeInverse(normalizer.mean, normalizer.std)
if attack_type=='Clean':
    trigger_path = os.path.join(args.trigger_dir, f'fixed_{patch_size}.npy')
    if not os.path.exists(trigger_path):
        trigger = Image.open('./trigger/fixed_trigger.png').convert('RGB')
        t_preprocess_fn = [transforms.Resize((patch_size, patch_size))]+ preprocess_fn.transforms[1:]
        t_transform = transforms.Compose(t_preprocess_fn)
        trigger = t_transform(trigger)
        np.save(trigger_path, trigger)
    else:
        trigger = np.load(trigger_path)
        trigger = torch.from_numpy(trigger)
else: # Ours
    # trigger_path = os.path.join(args.trigger_dir, f'On_{adversary_task}_Tgt_{target_cls}_L_{patch_size}.npy')
    # trigger = np.load(trigger_path)
    # trigger = torch.from_numpy(trigger)
    # # print("Trigger size:", trigger.shape) #* torch.Size([3, 22, 22])
    # # print("Trigger type:", type(trigger)) #* <class 'torch.Tensor'>

    triggers = []
    for i, ad_task in enumerate(adversary_task):
        trigger_path = os.path.join(args.trigger_dir, f'On_{ad_task}_Tgt_{target_cls[i]}_L_{patch_size}.npy')
        trigger = np.load(trigger_path)
        trigger = torch.from_numpy(trigger)
        triggers.append(trigger)


if len(adversary_task) == 1:
    applied_patch, mask, x_location, y_location = corner_mask_generation(trigger, image_size=(3, 224, 224))
    # applied_patch, mask, x_location, y_location = random_mask_generation(trigger, image_size=(3, 224, 224))
    # applied_patch, mask = distributed_corner_mask_generation(trigger, image_size=(3, 224, 224))
    # applied_patch, mask = distributed_random_mask_generation(trigger, image_size=(3, 224, 224))
else:
    applied_patch, mask = multi_mask_generation(triggers, image_size=(3, 224, 224))
    # applied_patch, mask = distributed_multi_mask_generation(triggers, image_size=(3, 224, 224))

applied_patch = torch.from_numpy(applied_patch)
mask = torch.from_numpy(mask)
print("Trigger size:", trigger.shape)
vutils.save_image(inv_normalizer(applied_patch), f"./src/vis_test/multi-patch.png")
# vutils.save_image(inv_normalizer(applied_patch), f"./src/vis_distributed_patch/{attack_type}_ap_seed0.png")

# exit()

### Log
args.logs_path = os.path.join(args.logs_dir, model)
str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
if not os.path.exists(args.logs_path):
    os.makedirs(args.logs_path)
# log = create_log_dir(args.logs_path, 'log_{}_task_arithmetic.txt'.format(str_time_))


### Model fusion
from ties_merging_utils import *
ft_checks = []
for dataset_name in exam_datasets:
    # clean model
    ckpt_name = os.path.join(args.save, dataset_name, 'finetuned.pt')
    # backdoored model
    if dataset_name in adversary_task: #! clean的话脚本里面没有添加adversary_task
        # 获取adversary_task的index
        index = adversary_task.index(dataset_name)
        ckpt_name = os.path.join(args.save, dataset_name+f'_On_{dataset_name}_Tgt_{target_cls[index]}_L_{patch_size}', 'finetuned.pt')
        # ckpt_name = os.path.join(args.save, dataset_name+f'_Dynamic_{adversary_task}_Tgt_{target_cls}_L_{patch_size}', 'finetuned.pt')
        # ckpt_name = os.path.join(args.save, dataset_name+f'_BadNets_{adversary_task}_Tgt_{target_cls}_L_{patch_size}', 'finetuned.pt')
    ft_checks.append(torch.load(ckpt_name).state_dict())
    print(ckpt_name)
ptm_check = torch.load(pretrained_checkpoint).state_dict() # 加载预训练基础模型

remove_keys = []
flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in ft_checks]) #! 微调模型向量
flat_ptm = state_dict_to_vector(ptm_check, remove_keys) #! 基础模型向量
tv_flat_checks = flat_ft - flat_ptm #! 各任务参数变化量
scaling_coef_ls = torch.ones((len(flat_ft)))*args.scaling_coef_  #! 缩放系数矩阵，TA中使用的是定值0.3
# scaling_coef_ls[0] = 0.0 #* ASR=96%

#TODO generate virtual model
# print(tv_flat_checks.shape) #* torch.Size([6, 113448705])
# print(type(tv_flat_checks)) #* <class 'torch.Tensor'>
# tv_flat_checks, selected_vector_id = generate_synthetic_task_vector(tv_flat_checks, num_synthetic=3)
# print("Selected vector id:", selected_vector_id)

# # 打印各任务参数变化量以及范数
# for i in range(len(tv_flat_checks)):
#     print(f"Task {i} norm: {torch.norm(tv_flat_checks[i])}")
#     print(f"Task {i} : {tv_flat_checks[i]} ")
#     # print(f"Task {i} norm: {torch.norm(tv_flat_checks[i])}")
    


print("Scaling coefs:", scaling_coef_ls)
# Scaling coefs: tensor([0.3000, 0.3000, 0.3000, 0.3000, 0.3000, 0.3000])

merged_check = flat_ptm
for i in range(len(tv_flat_checks)):
    merged_check = merged_check+scaling_coef_ls[i]*tv_flat_checks[i]
merged_state_dict = vector_to_state_dict(merged_check, ptm_check, remove_keys=remove_keys)
if use_merged_model:
    image_encoder.load_state_dict(merged_state_dict, strict=False)


### Evaluation
accs = []
asrs = []
asrs_crop = []

for dataset in exam_datasets:
    # clean
    if test_utility==True:
        metrics = eval_single_dataset(image_encoder, dataset, args) # can switch to eval_single_dataset_with_frozen_text_encoder
        accs.append(metrics.get('top1')*100)

    # backdoor #! badmergingON 只在target_task上做backdoor attack
    if test_effectiveness==True and dataset in target_task: 
        backdoored_cnt = 0
        non_target_cnt = 0
        backdoored_cnt_crop = 0
        non_target_cnt_crop = 0
        
        index = target_task.index(dataset)
        backdoor_info = {'mask': mask, 'applied_patch': applied_patch, 'target_cls': target_cls[index]}
        metrics_bd = eval_single_dataset(image_encoder, dataset, args, backdoor_info=backdoor_info) # can switch to eval_single_dataset_with_frozen_text_encoder
        print("*"*20, "Evaluate crop", "*"*20)
        metrics_bd_crop = eval_single_dataset(image_encoder, dataset, args, backdoor_info=backdoor_info, crop=True)
        backdoored_cnt += metrics_bd['backdoored_cnt']
        non_target_cnt += metrics_bd['non_target_cnt']
        backdoored_cnt_crop += metrics_bd_crop['backdoored_cnt']
        non_target_cnt_crop += metrics_bd_crop['non_target_cnt']
        asrs.append(backdoored_cnt/non_target_cnt)
        asrs_crop.append(backdoored_cnt_crop/non_target_cnt_crop)

### Metrics
if test_utility:
    print('Avg ACC:' + str(np.mean(accs)) + '%')

if test_effectiveness:
    print('Avg ASR:'+  str(np.mean(asrs)*100) + '%')
    print('Avg ASR Crop:'+  str(np.mean(asrs_crop)*100) + '%')

    # print('Backdoor acc:', backdoored_cnt/non_target_cnt)
    # print('Backdoor acc Crop:', backdoored_cnt_crop/non_target_cnt_crop)