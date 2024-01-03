import os, glob, json, argparse, json
import torch, clip
import faiss, nltk
import numpy as np
import pandas as pd
import webdataset as wds
from tqdm import tqdm
from logger import get_logger
from textblob import TextBlob
from nltk.util import everygrams
from sklearn.cluster import KMeans
from math import ceil

# TODO remove 'root' dependency, add open_clip, group_vit support, add multiprocessing support.

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root', type=str, required=True,
    )
    parser.add_argument(
        '--data', type=str, required=True, choices=['c3', 'c12', 'y14'],
    )
    parser.add_argument(
        '--index_data', type=str, choices=['c3', 'c12', 'y14'], default='c3'
    )
    parser.add_argument(
        '--size', type=int, default=8, help="for vision driven expansion", 
    )
    parser.add_argument(
        '--model', type=str, required=True, choices=['clip-vit-b-32', 'clip-vit-b-16', 'clip-vit-l-14'], 
    )
    parser.add_argument(
        '--batch', type=int, default=64, help="for processing textual inputs"
    )
    return parser.parse_args()


def cocu(
    image_idx: int,  
    l: int,  # - no. of retrieved images (L) in vision-driven expansion
    index,
    data_img_emb: np.ndarray,
    index_img_emb: np.ndarray,
    index_semantics,
    data_concept2idx,
    data_concept2emb, 
    enable_kmeans=False,
):
    '''
        len(index_semantics) - 2,751,755 for conceptual captions 3M (CC3M); 

        semantics - given an image caption 'alpaca resting in grass', semantics refers to a Python List ['alpaca', 'grass'].
    '''
    # anchor image. 
    img_emb = data_img_emb[[image_idx]]

    # search a small set of images similar to the anchor.
    _, local_img_idxs = index.search(img_emb, l)
    l_semantics = [index_semantics[idx] for idx in local_img_idxs[0]]
    l_img_emb = index_img_emb[local_img_idxs.squeeze(), :]

    # init mini concept archive.
    archive = []
    for semantics in l_semantics:
        if len(semantics) == 0:
            continue
        archive.extend(semantics.split(', '))
    archive = list(set(archive))
    if len(archive) == 0:
        return {}, {}
    
    local_probs = np.zeros(len(archive))
    global_probs = np.zeros(len(archive))
    max_prob = 0.0

    # concept curation
    archive_concept_emb = {}
    for concept_index, concept in enumerate(archive):
        global_concept_idx = data_concept2idx[concept]
        concept_emb = data_concept2emb[global_concept_idx, :]
        archive_concept_emb[concept] = concept_emb
        # reference images with respect to {concept}.
        r_img_idx = [idx for idx, semantics in enumerate(l_semantics) if concept in semantics]
        r_img_emb = l_img_emb[r_img_idx, :]
        # append anchor to tail.
        r_img_emb = np.concatenate([r_img_emb, img_emb])
        sim = concept_emb @ r_img_emb.T
        global_probs[concept_index] = sim[-1]
        local_probs[concept_index] = sim[-1] / np.average(sim)
        max_prob = max(sim.max(), max_prob)

    probs = global_probs / max_prob + local_probs
    scored_archive = {concept: prob for concept, prob in zip(archive, probs)}
    scored_archive = {k: v for k, v in sorted(scored_archive.items(), key=lambda item: item[1], reverse=True)}

    # cluster over scored archive.
    if enable_kmeans:
        kmeans_scored_archive = {}
        if len(scored_archive) > 2:
            archive_concept_emb = np.stack(
                [archive_concept_emb[key] for key in scored_archive.keys()], axis=0
            )
            archive_cluster_labels = KMeans(n_clusters=ceil(len(scored_archive)/2), random_state=0, n_init=5, max_iter=100).fit(archive_concept_emb).labels_
            # init cluster.
            for cluster_label in archive_cluster_labels:
                kmeans_scored_archive[str(cluster_label)] = {}
            # save cluster.
            for cluster_label, concept, score in zip(archive_cluster_labels, scored_archive.keys(), scored_archive.values()):
                kmeans_scored_archive[str(cluster_label)][concept] = score
    else:
        kmeans_scored_archive = {}
    return scored_archive, kmeans_scored_archive


def main():
    
    args = parse_args()
    logger = get_logger()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_ckpts = {
        'clip-vit-b-32': 'ViT-B/32',
        'clip-vit-b-16': 'ViT-B/16',
        'clip-vit-l-14': 'ViT-L/14',
    }

    # load clip model here.
    if args.model.startswith('clip'):
        model, _ = clip.load(clip_ckpts[args.model], device=device)
    else:
        raise NameError

    assert os.path.exists(f'{args.root}/output/{args.data}_emb/{args.model}/img_emb/img_emb_0.npy')
    assert os.path.exists(f'{args.root}/output/c3_emb/{args.model}/img_emb/img_emb_0.npy')
    assert os.path.exists(f'{args.root}/output/c3_index/{args.model}/img.index')

    logger.info(f"load CLIP image embeddings: \'{args.data}\'.")
    imgs_emb = np.load(f'{args.root}/output/{args.data}_emb/{args.model}/img_emb/img_emb_0.npy')  
    logger.info(f"load metadata: \'{args.data}\'.")
    meta = {path: idx for idx, path in enumerate(pd.read_parquet(f'{args.root}/output/{args.data}_metadata_0.parquet')['image_path'].to_list())}
    logger.info(f"load search index CLIP image embeddings: \'{args.index_data}\'.")
    index_imgs_emb = np.load(f'{args.root}/output/{args.index_data}_emb/{args.model}/img_emb/img_emb_0.npy')
    logger.info(f"load search index: \'{args.index_data}\'.")
    faiss_idx = faiss.read_index(f'{args.root}/output/{args.index_data}_index/{args.model}/img.index')
    
    # pre-load a gallery.
    data = json.load(open(f'{args.root}/rewrite/gallery.json'))
    gallery = []
    for item in data:
        gallery.extend(data[item])
    # avoid repetition.
    gallery_set = {}
    for concept in gallery:
        if concept not in gallery_set:
            gallery_set[concept.lower()] = None
    # get plurals for each concept.
    for concept in gallery_set.copy():
        words = TextBlob(concept).words
        words[-1] = words[-1].pluralize()
        if ' '.join(words) not in gallery_set:
            gallery_set[' '.join(words)] = None
    # extract nouns / phrases from captions.
    logger.info(f"load search index captions: \'{args.index_data}\'.")
    if os.path.exists(f'{args.root}/output/{args.index_data}_metadata_0.parquet'):
        index_meta = pd.read_parquet(f'{args.root}/output/{args.index_data}_metadata_0.parquet')
    else:
        index_meta = pd.read_parquet(f'{args.root}/output/{args.index_data}_emb/{args.model}/metadata/metadata_0.parquet')
        logger.info(f'extract on \'{args.data}\' image captions.')
        extracts = []
        for caption in tqdm(index_meta['caption'], leave=False):
            extract = []
            grams = [' '.join(gram) for gram in everygrams(nltk.word_tokenize(caption.lower()), max_len=4)]  # use 4 gram here.
            for gram in grams:
                if gram in gallery_set and gram not in extract:
                    extract.append(gram)
            extracts.append(', '.join(extract))
        # update search index metadata.
        index_meta['extracts'] = extracts
        # save it, only do the whole thing one time.
        index_meta.to_parquet(f'{args.root}/output/{args.index_data}_metadata_0.parquet')

    logger.info('inference CLIP embeddings for textual concepts.')
    # forward textual concepts to CLIP embeddings.
    texts = [f'a photo of a {concept}' for concept in list(gallery_set.keys())]
    data_concept2emb = []
    if args.model.startswith('clip'):
        text_tokens = clip.tokenize(texts).to(device)
        for batch_idx in tqdm(range(0, len(text_tokens), args.batch), leave=False):
            with torch.no_grad():
                batch_tokens = text_tokens[batch_idx: batch_idx + args.batch]
                batch_features = model.encode_text(batch_tokens)
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                batch_features = batch_features.cpu().numpy()
                data_concept2emb.append(
                    batch_features
                )
        data_concept2emb = np.concatenate(data_concept2emb, axis=0)
    else:
        raise NameError(f'{args.model} not supported.')
    
    # rewrite semantics using concept curation.
    data_concept2idx = {name: idx for idx, name in enumerate(gallery_set.keys())}
    index_names = index_meta['extracts'].to_list()

    logger.info(f'save re-written tarfiles in \'{args.root}/local_data/{args.model}_{args.size}_{args.data}_shards\'.')
    if not os.path.exists(f'{args.root}/local_data/{args.model}_{args.size}_{args.data}_shards'):
        os.mkdir(f'{args.root}/local_data/{args.model}_{args.size}_{args.data}_shards')

    pattern = f'{args.root}/local_data/{args.model}_{args.size}_{args.data}_shards/{args.data}-%05d.tar'
    len_itp = len(glob.glob(f'{args.root}/local_data/{args.data}_shards_256x256/*.tar')) - 1  # start with 00000.
    wds_simpleshardlist = f'{args.root}/local_data/{args.data}_shards_256x256/' + args.data + '-{' + f'00000..{str(len_itp).zfill(5)}' + '}.tar'
    src = wds.DataPipeline(
        wds.SimpleShardList(wds_simpleshardlist),
        wds.tarfile_to_samples(),
        wds.decode('pil'),
        wds.to_tuple(
            "__key__", 
            "jpg;png;jpeg", 
            "txt;text", 
        ),
        wds.map_tuple(None, None, None)
    )

    logger.info(f'rewrite begins. this may take for a while.')
    with wds.ShardWriter(
        pattern,
        maxcount=1e4
    ) as dst:
        for image_idx, (key, image, text) in tqdm(enumerate(src), leave=False):
            image_idx = meta[key]
            cocu_sample, kmeans_cocu_sample = cocu(
                image_idx,
                args.size,
                faiss_idx,
                imgs_emb,
                index_imgs_emb,
                index_names,
                data_concept2idx,
                data_concept2emb,
            )
            sample = {
                '__key__': key,
                "jpg": image,
                "txt": text + '\n' + json.dumps(cocu_sample) + '\n' + json.dumps(kmeans_cocu_sample)
            }
            dst.write(sample)

if __name__ == '__main__':
    main()

