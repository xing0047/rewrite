## Overview

Your data should look like:

```
CoCu
├── $DATASETS
│   ├── c3_shards
│   ├── c12_shards
│   ├── y14_shards
│   ├── imagenet_shards
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2012
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClass
│   │   │   ├── ImageSets
│   │   │   │   ├── Segmentation
│   │   ├── VOC2010
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClassContext
│   │   │   ├── ImageSets
│   │   │   │   ├── SegmentationContext
│   │   │   │   │   ├── val.txt
│   │   │   ├── trainval_merged.json
│   ├── ade
│   │   ├── ADEChallengeData2016
│   │   │   ├── annotations
│   │   │   │   ├── validation
│   │   │   ├── images
│   │   │   │   ├── validation
│   ├── coco_stuff164k
│   │   ├── images
│   │   │   ├── val2017
│   │   ├── annotations
│   │   │   ├── val2017
│   ├── ImageNetS
│   │   ├── ImageNetS300
│   │   │   ├── validation
│   │   │   ├── validation-segmentation
│   │   ├── ImageNetS50
│   │   │   ├── validation
│   │   │   ├── validation-segmentation
│   ├── coco
│   │   ├── images
│   │   │   ├── val2017
│   │   ├── annotations
│   │   │   ├── val2017
```

## Pre-train Data

Setting a lower `processes_count` (in my practice is `4`) ensures higher successful rate of img2dataset downloads. If still got a low successful rate, please refer to [img2dataset](https://github.com/rom1504/img2dataset).

#### GCC3M

Please download CC3M metadata [here](https://storage.cloud.google.com/gcc-data/Train/GCC-training.tsv?_ga=2.191230122.-1896153081.1529438250) and name it to `c3.tsv`. Then run script below.
```
img2dataset --url_list c3.tsv \
  --input_format "tsv" \
  --url_col "url" \
  --caption_col "caption" \
  --output_format webdataset \
  --output_folder $DATASETS/c3_shards \
  --processes_count 4 \
  --thread_count 16 \
  --image_size 512 \
  --resize_mode keep_ratio \
  --resize_only_if_bigger True \
  --enable_wandb False
rename -d 's/^/c3-/' ${DATASETS}/c3_shards/*
```

#### GCC12M

Please download CC12M data [here](https://github.com/google-research-datasets/conceptual-12m) and name it to `c12.tsv`. Then run script below.
```
img2dataset --url_list c12.tsv \
  --input_format "tsv" \
  --url_col "url" \
  --caption_col "caption" \
  --output_format webdataset \
  --output_folder $DATASETS/c12_shards \
  --processes_count 4 \
  --thread_count 16 \
  --image_size 512 \
  --resize_mode keep_ratio \
  --resize_only_if_bigger True \
  --enable_wandb False
rename -d 's/^/c12-/' ${DATASETS}/c12_shards/*
```

#### YFCC14M

First, download YFCC14M subset as suggested [here](https://github.com/openai/CLIP/blob/main/data/yfcc100m.md). 
```
wget https://openaipublic.azureedge.net/clip/data/yfcc100m_subset_data.tsv.bz2
bunzip2 yfcc100m_subset_data.tsv.bz2
```

Next, run preprocessing script to derive `yfcc100m_dataset.sql` and an annotation file `yfcc14m_dataset.tsv`.
```
python convert_dataset/create_subset.py --input-dir . --output-dir . --subset yfcc100m_subset_data.tsv
```

Then download YFCC14M dataset. Please make sure there is enough disk space to do this.
```
pip install git+https://gitlab.com/jfolz/yfcc100m.git
mkdir -p yfcc100m_meta
python -m yfcc100m.convert_metadata . -o yfcc100m_meta --skip_verification
mkdir -p yfcc100m_zip
python -m yfcc100m.download yfcc100m_meta -o yfcc100m_zip
```

Last, convert this to webdataset format.
```
python convert_dataset/convert_yfcc14m.py --root yfcc100m_zip --info yfcc14m_dataset.tsv --shards y14_shards
rename -e 's/yfcc14m-0/y14-/g' local_data/y14_shards/*
```

## Evaluation Data - Segmentation

#### Pascal VOC

Please download Pascal VOC 2012 as follows.
```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```

#### Pascal Context

Please download train and val set of Pascal Context as below.

```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
```

Additionally, get `trainval_merged.json` from [here](https://codalabuser.blob.core.windows.net/public/trainval_merged.json) and put it under `$DATASETS/VOCdevkit/VOC2010/`. Then run as follows,

```
python tools/convert_datasets/pascal_context.py $DATASETS/VOCdevkit $DATASETS/VOCdevkit/VOC2010/trainval_merged.json
```
#### COCO Stuff

```
wget http://images.cocodataset.org/zips/val2017.zip
wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip
unzip val2017.zip -d images/
unzip stuffthingmaps_trainval2017.zip -d annotations/
python tools/convert_datasets/coco_stuff164k.py $DATASETS/coco_stuff164k --nproc 8
```

#### ADE20K
```
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
```

#### Cityscapes
The Cityscapes dataset could be downloaded from [here](https://www.cityscapes-dataset.com/login/) after registration. After download, convert this data by:
```
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
```

#### ImageNet-S

First, you should have a copy of ImageNet dataset. A way to do this is to download from [here](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data). Save or soft-link it to `$DATASETS/ImageNet`.

Then, follow instructions given [here](https://github.com/LUSSeg/ImageNet-S). 
```
git clone https://github.com/LUSSeg/ImageNet-S.git
cd ImageNet-S
```

We only need validation set in our setting. Prepare for ImageNet-S-50 as follows:
```
python datapreparation_val.py \
  --imagenet-dir $DATASETS/ImageNet \
  --save-dir $DATASETS/ImageNet-S \
  --mode 50
```
Samely, for ImageNet-S-300:
```
python datapreparation_val.py \
  --imagenet-dir $DATASETS/ImageNet \
  --save-dir $DATASETS/ImageNet-S \
  --mode 300
```

## Evaluation Data - Classification

#### ImageNet


