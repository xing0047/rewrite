# Rewrite Caption Semanitcs: Bridging Semantic Gaps for Language Supervised Semantic Segmentation

This is the official repository of the following paper:
> **[Rewrite Caption Semantics: Bridging Semantic Gaps for Language Supervised Semantic Segmentation](https://arxiv.org/abs/2309.13505)**<br>
> NeurIPS 2023<br>
> Yun Xing, Jian Kang, Aoran Xiao, Jiahao Nie, Ling Shao, Shijian Lu<br>

> **<p align="justify"> Abstract:** *Vision-Language Pre-training has demonstrated its remarkable zero-shot recognition ability and poten-tial to learn generalizable visual representations from language supervision. Taking a step ahead, language-supervised semantic segmentation enables spatial localization of textual inputs by learning pixel grouping solely from image-text pairs. Nevertheless, the state-of-the-art suffers from clear semantic gaps between visual and textual modality: plenty of visual concepts appeared in images are missing in their paired captions. Such semantic misalignment circulates in pre-training, leading to inferior zero-shot performance in dense predictions due to insufficient visual concepts captured in textual representations. To close such semantic gap, we propose Concept Curation (CoCu), a pipeline that leverages CLIP to compensate for the missing semantics. For each image-text pair, we establish a concept archive that maintains potential visually-matched concepts with our proposed vision-driven expansion and text-to-vision-guided ranking. Relevant concepts can thus be identified via cluster-guided sampling and fed into pre-training, thereby bridging the gap between visual and textual semantics. Extensive experiments over a broad suite of 8 segmentation benchmarks show that CoCu achieves superb zero-shot transfer performance and greatly boosts language-supervised segmentation baseline by a large margin, suggesting the value of bridging semantic gap in pre-training data.* </p>

## Updates

- [09/2023] arXiv available.

## Environmental Setup
```
conda create -n rewrite python=3.7 -y
conda activate rewrite
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
