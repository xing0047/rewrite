# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

custom_imports = dict(
    imports=[
        'segmentation.datasets.coco_object', 
        'segmentation.datasets.pascal_voc', 
        'segmentation.datasets.imagenets', 
        'segmentation.datasets.ade', 
        'segmentation.datasets.cityscapes', 
        'segmentation.datasets.coco_stuff'
    ], 
    allow_failed_imports=False
)

