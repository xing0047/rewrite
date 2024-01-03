ckpt=/path/to/ckpt
data='voc'  # ['voc', 'pcon', 'coco', 'in50', 'in300', 'city', 'ade', 'stuff']
./tools/dist_launch.sh main_group_vit.py configs/eval/eval_${data}.yml 8 --resume ${ckpt}