# [NeurIPS 2023] Bridging Semantic Gaps for Language Supervised Semantic Segmentation

### [[Paper]()]

This is the official repository of the following paper.
> **Bridging Semantic Gaps for Language Supervised Semantic Segmentation**<br>
> [Yun Xing](https://scholar.google.com/citations?user=uOAYTXoAAAAJ&hl=en&oi=ao), [Jian Kang](https://www.linkedin.com/in/alan-kang-6497b5239), [Aoran Xiao](https://scholar.google.com/citations?user=yGKsEpAAAAAJ&hl=en), [Jiahao Nie](https://niejiahao1998.github.io/), [Ling Shao](https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=zh-CN&oi=ao), [Shijian Lu](https://scholar.google.com/citations?user=uYmK-A0AAAAJ&hl=en&oi=ao)<br>

Code will be released soon. Please stay tuned.

## Updates

- [09/2023] arXiv available.

## TODO
- [ ] release visualization code.
- [ ] release curation code.

## Environmental Setup
```
conda create -n cocu python=3.7 -y
conda activate cocu
```
##### torch
```
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```
##### apex
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
##### mmcv
```
pip install mmcv-full==1.3.14 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
```
##### others
```
pip install -r requirements.txt
```
##### clip-retrieval
```
pip install clip-retrieval
```

## Citation

Please consider citing our paper if you find our work useful for you.

## Acknowledgement

The repo is built on [GroupViT](https://github.com/NVlabs/GroupViT) and [clip-retrieval](https://github.com/rom1504/clip-retrieval).
