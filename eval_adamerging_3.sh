CUDA_VISIBLE_DEVICES=2 python3 src/main_adamerging_badmergingon.py --attack-type 'Clean' --adversary-task '' --target-task 'CIFAR100' \
	--target-cls 1 --patch-size 100 --alpha 5 --test-utility --test-effectiveness True # clean-on

CUDA_VISIBLE_DEVICES=2 python3 src/main_adamerging_badmergingon.py --attack-type 'On' --adversary-task 'CIFAR100' --target-task 'CIFAR100' \
	--target-cls 1 --patch-size 100 --alpha 5 --test-utility --test-effectiveness True # badmerging-on

# CUDA_VISIBLE_DEVICES=2 python3 src/main_adamerging_badmergingoff.py --attack-type 'Clean' --adversary-task '' --target-task 'Cars' \
#     --target-cls 1 --patch-size 100 --alpha 5 --test-utility --test-effectiveness True --num-shadow-data 10 --num-shadow-classes 300 # clean-off

CUDA_VISIBLE_DEVICES=2 python3 src/main_adamerging_badmergingoff.py --attack-type 'Off' --adversary-task 'CIFAR100' --target-task 'Cars' \
    --target-cls 1 --patch-size 100 --alpha 5 --test-utility --test-effectiveness True --num-shadow-data 10 --num-shadow-classes 300 # badmerging-off


# patch size 22=1%  50=5%  71=10%  87=15%   100=20%   123=30%     