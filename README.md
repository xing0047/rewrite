# [NeurIPS 2023] Bridging Semantic Gaps for Language Supervised Semantic Segmentation

This is the official repository of the following paper.
> **Bridging Semantic Gaps for Language Supervised Semantic Segmentation**<br>
> [Yun Xing](https://scholar.google.com/citations?user=uOAYTXoAAAAJ&hl=en&oi=ao), [Jian Kang](https://www.linkedin.com/in/alan-kang-6497b5239), [Aoran Xiao](https://scholar.google.com/citations?user=yGKsEpAAAAAJ&hl=en), [Jiahao Nie](https://niejiahao1998.github.io/), [Ling Shao](https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=zh-CN&oi=ao), [Shijian Lu](https://scholar.google.com/citations?user=uYmK-A0AAAAJ&hl=en&oi=ao)<br>

## Updates

- [09/2023] Arxiv available.
- [09/2023] Code released.

## Environmental Setup
```
conda create -n cocu python=3.7 -y
conda activate cocu
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```
##### install apex
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Result

## Data Preparation

## Train

## Inference

## Variants

## FAQ

## Citation

Please consider citing our paper if you find our work useful.

## Acknowledgement

The repo is built on [GroupViT](https://github.com/NVlabs/GroupViT) and [clip-retrieval](https://github.com/rom1504/clip-retrieval).
