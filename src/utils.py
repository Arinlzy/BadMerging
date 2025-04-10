import os
import random
import torch
import pickle
import math
import numpy as np
import torchvision   

class NormalizeInverse(torchvision.transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

def multi_mask_generation(patch=None, image_size=(3, 224, 224)):
    num_masks = len(patch)
    applied_patch = np.zeros(image_size)
    patch_h, patch_w = patch[0].shape[1], patch[0].shape[2]
    
    # 计算每个mask的随机位置
    locations = []
    for _ in range(num_masks):
        x_location = random.randint(0, image_size[1] - patch_h)
        y_location = random.randint(0, image_size[2] - patch_w)
        locations.append((x_location, y_location))
    
    # 将patch应用到每个位置
    for i, (x_location, y_location) in enumerate(locations):
        applied_patch[:, x_location:x_location + patch_h, y_location:y_location + patch_w] = patch[i]
    
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    
    return applied_patch, mask


def corner_mask_generation(patch=None, image_size=(3, 224, 224)):
    applied_patch = np.zeros(image_size)
    x_location = image_size[1]-patch.shape[1]
    y_location = image_size[2]-patch.shape[2]
    applied_patch[:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]] = patch
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return applied_patch, mask, x_location, y_location

def random_mask_generation(patch=None, image_size=(3, 224, 224)):
    applied_patch = np.zeros(image_size)
    x_location = np.random.randint(0, image_size[1] - patch.shape[1])
    y_location = np.random.randint(0, image_size[2] - patch.shape[2])
    applied_patch[:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]] = patch
    mask = applied_patch.copy()
    mask[mask!= 0] = 1.0
    return applied_patch, mask, x_location, y_location

def distributed_corner_mask_generation(patch=None, image_size=(3, 224, 224)):
    applied_patch = np.zeros(image_size)
    patches = torch.chunk(patch, 2, dim=2)  # 沿高度维度切分成2份
    patches = [torch.chunk(p, 2, dim=1) for p in patches]  # 沿宽度维度再切分
    
    patch_h, patch_w = patches[0][0].shape[1], patches[0][0].shape[2]
    
    # 计算四个角的位置
    locations = [
        (0, 0),  # 左上角
        (0, image_size[2] - patch_w),  # 右上角
        (image_size[1] - patch_h, 0),  # 左下角
        (image_size[1] - patch_h, image_size[2] - patch_w)  # 右下角
    ]
    
    for (x, y), sub_patch in zip(locations, [p[0] for p in patches] + [p[1] for p in patches]):
        applied_patch[:, x:x + patch_h, y:y + patch_w] = sub_patch.numpy()
    
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    
    return applied_patch, mask

def distributed_random_mask_generation(patch=None, image_size=(3, 224, 224), seed=0):

    np.random.seed(seed)

    applied_patch = np.zeros(image_size)
    patches = torch.chunk(patch, 2, dim=2)  # 沿高度维度切分成2份
    patches = [torch.chunk(p, 2, dim=1) for p in patches]  # 沿宽度维度再切分
    
    patch_h, patch_w = patches[0][0].shape[1], patches[0][0].shape[2]
    
    # 生成随机位置
    locations = []
    for _ in range(4):
        x = random.randint(0, image_size[1] - patch_h)
        y = random.randint(0, image_size[2] - patch_w)
        locations.append((x, y))
    
    for (x, y), sub_patch in zip(locations, [p[0] for p in patches] + [p[1] for p in patches]):
        applied_patch[:, x:x + patch_h, y:y + patch_w] = sub_patch.numpy()
    
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    
    return applied_patch, mask

def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def torch_load_old(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def torch_save(model, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.cpu(), save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path)
    if device is not None:
        model = model.to(device)
    return model



def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, 'to'):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, 'predict_proba'):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
