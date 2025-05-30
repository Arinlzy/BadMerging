import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import time
import sys
# import numpy as np
sys.path.append('./src')
sys.path.append('.')
from eval import eval_single_dataset
from args import parse_arguments
from utils import *     
from PIL import Image
import torchvision.utils as vutils
import torchvision.transforms as transforms

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
patch_size = args.patch_size
alpha = args.alpha
num_shadow_data = args.num_shadow_data
num_shadow_classes = args.num_shadow_classes
test_utility = args.test_utility
test_effectiveness = args.test_effectiveness

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
    trigger_path = os.path.join(args.trigger_dir, f"Off_{target_task}_Tgt_{target_cls}_SD_{num_shadow_data}_SC_{num_shadow_classes}_L_{patch_size}.npy")
    trigger = np.load(trigger_path)
    trigger = torch.from_numpy(trigger)

applied_patch, mask, x_location, y_location = corner_mask_generation(trigger, image_size=(3, 224, 224))
# applied_patch, mask, x_location, y_location = random_mask_generation(trigger, image_size=(3, 224, 224))
# applied_patch, mask = distributed_corner_mask_generation(trigger, image_size=(3, 224, 224))
# applied_patch, mask = distributed_random_mask_generation(trigger, image_size=(3, 224, 224))

applied_patch = torch.from_numpy(applied_patch)
mask = torch.from_numpy(mask)
print("Trigger size:", trigger.shape)
# vutils.save_image(inv_normalizer(applied_patch), f"./src/vis_distributed_patch/{attack_type}_ap_seed0.png")

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
    if dataset_name==adversary_task:
        ckpt_name = os.path.join(args.save, dataset_name+f'_Off_{target_task}_Tgt_{target_cls}_SD_{num_shadow_data}_SC_{num_shadow_classes}_L_{patch_size}', 'finetuned.pt')
    ft_checks.append(torch.load(ckpt_name).state_dict())
    print(ckpt_name)
ptm_check = torch.load(pretrained_checkpoint).state_dict()
check_parameterNamesMatch(ft_checks + [ptm_check])

# flat
remove_keys = []
print(f"Flattening out Checkpoints")
flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in ft_checks])
flat_ptm = state_dict_to_vector(ptm_check, remove_keys)
tv_flat_checks = flat_ft - flat_ptm
assert check_state_dicts_equal(vector_to_state_dict(flat_ptm, ptm_check, remove_keys), ptm_check)
assert all([check_state_dicts_equal(vector_to_state_dict(flat_ft[i], ptm_check, remove_keys), ft_checks[i])for i in range(len(ft_checks))])

#TODO generate virtual model
# print(tv_flat_checks.shape) #* torch.Size([6, 113448705])
# print(type(tv_flat_checks)) #* <class 'torch.Tensor'>


# merging
K = 20
merge_func = "dis-sum"
scaling_coef_ = args.scaling_coef_
print("Scaling coef:", scaling_coef_)

merged_tv = ties_merging(tv_flat_checks, reset_thresh=K, merge_func=merge_func,)
merged_check = flat_ptm + scaling_coef_ * merged_tv
merged_state_dict = vector_to_state_dict(merged_check, ptm_check, remove_keys=remove_keys)
image_encoder.load_state_dict(merged_state_dict, strict=False)


### Evaluation
accs = []
backdoored_cnt = 0
non_target_cnt = 0
backdoored_cnt_crop = 0
non_target_cnt_crop = 0
for dataset in exam_datasets:
    # clean
    if test_utility==True:
        metrics = eval_single_dataset(image_encoder, dataset, args)
        accs.append(metrics.get('top1')*100)

    # backdoor
    if test_effectiveness==True and dataset==target_task:
        backdoor_info = {'mask': mask, 'applied_patch': applied_patch, 'target_cls': target_cls}
        metrics_bd = eval_single_dataset(image_encoder, dataset, args, backdoor_info=backdoor_info)
        print("*"*20, "Evaluate crop", "*"*20)
        metrics_bd_crop = eval_single_dataset(image_encoder, dataset, args, backdoor_info=backdoor_info, crop=True)
        backdoored_cnt += metrics_bd['backdoored_cnt']
        non_target_cnt += metrics_bd['non_target_cnt']
        backdoored_cnt_crop += metrics_bd_crop['backdoored_cnt']
        non_target_cnt_crop += metrics_bd_crop['non_target_cnt']

### Metrics
if test_utility:
    print('Avg ACC:' + str(np.mean(accs)) + '%')

if test_effectiveness:
    print('Backdoor acc:', backdoored_cnt/non_target_cnt)
    print('Backdoor acc Crop:', backdoored_cnt_crop/non_target_cnt_crop)
