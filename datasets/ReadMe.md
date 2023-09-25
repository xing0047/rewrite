## Pre-train Data

As issue suggested, setting a lower `processes_count` (in my practice is `4`) ensures higher successful rate of img2dataset download. 

#### GCC3M

Please download [CC3M training split file](https://storage.cloud.google.com/gcc-data/Train/GCC-training.tsv?_ga=2.191230122.-1896153081.1529438250) and name it to `c3.tsv`. Then run script below.
```
img2dataset --url_list c3.tsv \
  --input_format "tsv" \
  --url_col "url" \
  --caption_col "caption" \
  --output_format webdataset \
  --output_folder local_data/c3_shards \
  --processes_count 4 \
  --thread_count 16 \
  --image_size 512 \
  --resize_mode keep_ratio \
  --resize_only_if_bigger True \
  --enable_wandb False
```

#### GCC12M

Please download CC12M data [here](https://github.com/google-research-datasets/conceptual-12m) and name it to `c12.tsv`. Then run script below.
```
img2dataset --url_list c12.tsv \
  --input_format "tsv" \
  --url_col "url" \
  --caption_col "caption" \
  --output_format webdataset \
  --output_folder local_data/c3_shards \
  --processes_count 4 \
  --thread_count 16 \
  --image_size 512 \
  --resize_mode keep_ratio \
  --resize_only_if_bigger True \
  --enable_wandb False
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
python convert_dataset/convert_yfcc14m.py --root yfcc100m_zip --info yfcc14m_dataset.tsv --shards yfcc14m_shards
```

## Evaluation Data

#### Pascal VOC

#### Pascal Context

#### ADE20K

#### Cityscapes

#### ImageNet-S

#### COCO Stuff
