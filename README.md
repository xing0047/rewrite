# Rewrite Caption Semantics: Bridging Semantic Gaps for Language Supervised Semantic Segmentation

This is the official repository of the following paper:
> **[Rewrite Caption Semantics: Bridging Semantic Gaps for Language Supervised Semantic Segmentation](https://openreview.net/forum?id=9iafshF7s3)**<br>
> NeurIPS 2023<br>
> Yun Xing, Jian Kang, Aoran Xiao, Jiahao Nie, Ling Shao, Shijian Lu<br>

> <p align="justify"> *Vision language pre-training has demonstrated its remarkable zero-shot recognition ability and potential to learn generalizable visual representations from language supervision. Taking a step ahead, language-supervised semantic segmentation enables spatial localization of textual inputs by learning pixel grouping solely from image-text pairs. Nevertheless, the state-of-the-art suffers from clear semantic gaps between visual and textual modality: plenty of visual concepts appeared in images are missing in their paired captions. Such semantic misalignment circulates in pre-training, leading to inferior zero-shot performance in dense predictions due to insufficient visual concepts captured in textual representations. To close such semantic gap, we propose Concept Curation (CoCu), a pipeline that leverages CLIP to compensate for the missing semantics. For each image-text pair, we establish a concept archive that maintains potential visually-matched concepts with our proposed vision-driven expansion and text-to-vision-guided ranking. Relevant concepts can thus be identified via cluster-guided sampling and fed into pre-training, thereby bridging the gap between visual and textual semantics. Extensive experiments over a broad suite of 8 segmentation benchmarks show that CoCu achieves superb zero-shot transfer performance and greatly boosts language-supervised segmentation baseline by a large margin, suggesting the value of bridging semantic gap in pre-training data.* </p>

## Updates

- [x] code released.
- [x] paper available.

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

## Run
#### Curation
```python
CONFIG='configs/train/cocu_clip-vit-b-16_8_c3_30e.yml'
DATA='c3'
MODEL='clip-vit-b-16'
```

##### Turn a set of image-caption pairs to CLIP embeddings.
```bash
bash scripts/inference.sh --data ${DATA} --model ${MODEL}
```

##### Take CLIP embeddings and make a search index out of it.
```bash
bash scripts/index.sh --data ${DATA} --model ${MODEL}
```

##### Rewrite semantics of image captions.
```bash
python rewrite/curation.py --data ${DATA} --model ${MODEL}
```

#### Pre-train
```bash
./tools/dist_launch.sh main_group_vit.py ${CONFIG} 4
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

The repo is built on [GroupViT](https://github.com/NVlabs/GroupViT) and [clip-retrieval](https://github.com/rom1504/clip-retrieval).
