import argparse
import json
import os
import os.path as osp
import random
import sys
import tarfile
import io
import numpy as np
import pandas as pd
import webdataset as wds
from tqdm import tqdm
from PIL import Image, ImageOps
import mmcv

def augment(
    image: Image.Image
):
    '''
        resize to [256, 256] with border.
    '''
    resized_image = None

    return resized_image

def write_dataset(args):

    pattern = os.path.join('/home/xingyun/GroupViT/local_data/y14_shards_256x256', f'y14-%05d.tar')
    with wds.ShardWriter(
        pattern,
        maxsize=int(args.maxsize)
    ) as sink:
        sink.verbose = 0

        tar_files = list(mmcv.scandir(args.root, suffix="tar"))
        image = text = None
        for idx, file in tqdm(enumerate(tar_files), desc="total", total=len(tar_files)):
            with tarfile.open(osp.join(args.root, file), "r") as tfile:  
                filename_list = tfile.getnames()
                for filename in tqdm(
                    filename_list,
                    position=1,
                    desc=f'{file}',
                    leave=None
                ):
                    if filename.endswith('.jpg'):
                        tarinfo = tfile.getmember(filename)
                        image = tfile.extractfile(tarinfo).read()
                        image = Image.open(io.BytesIO(image)).convert('RGB')
                        image = ImageOps.pad(image, (256, 256), color=(255, 255, 255))
                    elif filename.endswith('.text'):
                        tarinfo = tfile.getmember(filename)
                        text = tfile.extractfile(tarinfo).read().decode('utf-8')
                    else:
                        raise NameError(f'{osp.splitext(filename)[-1]} not supported')
                    
                    if image and text:
                        xkey = osp.splitext(filename)[0]
                        sample = {
                            '__key__': xkey,
                            'jpg': image,
                            'text': text
                        }
                        # write sample to sharded tar archives.    
                        sink.write(sample)
                        # clear sample
                        image = text = None

def parse_args():
    parser = argparse.ArgumentParser(
        """Generate sharded dataset from original ImageNet data.""")
    parser.add_argument('--maxsize', type=float, default=1e9)
    parser.add_argument('--maxcount', type=float, default=100000)
    parser.add_argument('--shards', help='directory where shards are written')
    parser.add_argument('--root', help='data root path')
    parser.add_argument('--info', help='tsv path')
    args = parser.parse_args()

    assert args.maxsize > 10000000
    assert args.maxcount < 1000000
    return args

def main():
    args = parse_args()

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    write_dataset(args=args)

if __name__ == '__main__':
    main()