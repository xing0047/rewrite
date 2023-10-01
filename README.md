# [NeurIPS 2023] Bridging Semantic Gaps for Language Supervised Semantic Segmentation

This is the official repository of the following paper.
> **[Bridging Semantic Gaps for Language Supervised Semantic Segmentation](https://arxiv.org/abs/2309.13505)**<br>
> [Yun Xing](https://scholar.google.com/citations?user=uOAYTXoAAAAJ&hl=en&oi=ao), [Jian Kang](https://www.linkedin.com/in/alan-kang-6497b5239), [Aoran Xiao](https://scholar.google.com/citations?user=yGKsEpAAAAAJ&hl=en), [Jiahao Nie](https://niejiahao1998.github.io/), [Ling Shao](https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=zh-CN&oi=ao), [Shijian Lu](https://scholar.google.com/citations?user=uYmK-A0AAAAJ&hl=en&oi=ao)<br>


```
Vision-Language Pre-training has demonstrated its remarkable zero-shot recognition ability and potential to learn generalizable visual representations from language supervision. Taking a step ahead, language-supervised semantic segmentation enables spatial localization of textual inputs by learning pixel grouping solely from image-text pairs. Nevertheless, the state-of-the-art suffers from clear semantic gaps between visual and textual modality: plenty of visual concepts appeared in images are missing in their paired captions. Such semantic misalignment circulates in pre-training, leading to inferior zero-shot performance in dense predictions due to insufficient visual concepts captured in textual representations. To close such semantic gap, we propose Concept Curation (CoCu), a pipeline that leverages CLIP to compensate for the missing semantics. For each image-text pair, we establish a concept archive that maintains potential visually-matched concepts with our proposed vision-driven expansion and text-to-vision-guided ranking. Relevant concepts can thus be identified via cluster-guided sampling and fed into pre-training, thereby bridging the gap between visual and textual semantics. Extensive experiments over a broad suite of 8 segmentation benchmarks show that CoCu achieves superb zero-shot transfer performance and greatly boosts language-supervised segmentation baseline by a large margin, suggesting the value of bridging semantic gap in pre-training data.
```

Code will be released soon. Please stay tuned.

## Updates

- [09/2023] arXiv available.
- [09/2023] accepted by [NeurIPS 2023](https://nips.cc/).

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
## Data Preparation

Please refer to [here](https://github.com/xing0047/CoCu/tree/main/data).

## Citation

Please consider citing our paper if you find our work useful for you.
```
@misc{xing2023bridging,
      title={Bridging Semantic Gaps for Language-Supervised Semantic Segmentation}, 
      author={Yun Xing and Jian Kang and Aoran Xiao and Jiahao Nie and Shao Ling and Shijian Lu},
      year={2023},
      eprint={2309.13505},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement

The repo is built on [GroupViT](https://github.com/NVlabs/GroupViT) and [clip-retrieval](https://github.com/rom1504/clip-retrieval).
