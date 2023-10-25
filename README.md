# Rewrite Caption Semanitcs: Bridging Semantic Gaps for Language Supervised Semantic Segmentation

This is the official repository of the following paper:
> **[Rewrite Caption Semantics: Bridging Semantic Gaps for Language Supervised Semantic Segmentation](https://arxiv.org/abs/2309.13505)**<br>
> NeurIPS 2023<br>
> Yun Xing, Jian Kang, Aoran Xiao, Jiahao Nie, Ling Shao, Shijian Lu<br>

We propose **Concept Curation** to rewrite caption semantics by leveraging a pre-trained vision-language model.

## Updates

- [09/2023] arXiv available.

## Environmental Setup
```
conda create -n cocu python=3.7 -y
conda activate cocu
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install mmcv-full==1.3.14 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
pip install -r requirements.txt
git clone https://github.com/ptrblck/apex.git
cd apex & pip install -v --no-cache-dir ./
```

## Data Preparation

Please refer to [data preparation page](https://github.com/xing0047/CoCu/tree/main/data).

## Run
#### Pre-train
```
./tools/dist_launch.sh main_group_vit.py /path/to/config 4
```

## Citation

Please consider citing our paper if you find our work useful.
```
@inproceedings{xing2023rewrite,
    title={Rewrite Caption Semantics: Bridging Semantic Gaps for Language-Supervised Semantic Segmentation}, 
    author={Yun Xing and Jian Kang and Aoran Xiao and Jiahao Nie and Shao Ling and Shijian Lu},
    booktitle={Advances in Neural Information Processing Systems},
    year={2023},
}
```

## Acknowledgement

The repo is built on [GroupViT](https://github.com/NVlabs/GroupViT).
