CUDA_VISIBLE_DEVICES=5 python3 src/ut_badmergingon.py --adversary-task 'EuroSAT' --model 'ViT-B-32' --target-cls 1 --mask-length 22 # Get universal trigger

CUDA_VISIBLE_DEVICES=5 python3 src/finetune_backdoor_badmergingon.py --adversary-task 'EuroSAT' --model 'ViT-B-32' --target-cls 1 --patch-size 22 --alpha 5 # BadMerging-on


# patch size 22=1%  50=5%  71=10%  87=15%   100=20%   123=30%     