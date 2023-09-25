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

Please download [CC12M training split file](https://storage.cloud.google.com/gcc-data/Train/GCC-training.tsv?_ga=2.191230122.-1896153081.1529438250) and name it to `c12.tsv`. Then run script below.
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

## Evaluation Data

#### Pascal VOC

#### Pascal Context

#### ADE20K

#### Cityscapes

#### ImageNet-S

#### COCO Stuff
