root='/home/xingyun/rewrite'

# data: c3
# model: clip-vit-b-32
CUDA_VISIBLE_DEVICES=7 clip-retrieval inference.worker \
	--tasks='[0]' \
	--input_dataset=${root}'/local_data/c3_shards_256x256/c3-{00000..00331}.tar' \
	--clip_model='ViT-B/32' \
	--output_folder=${root}'/output/c3_emb/clip-vit-b-32' \
	--batch_size 4096 \
	--input_format='webdataset' \
	--output_partition_count='1'

# data: c3
# model: clip-vit-b-16
# CUDA_VISIBLE_DEVICES=2 clip-retrieval inference.worker \
# 	--tasks='[0]' \
# 	--input_dataset=${root}'/local_data/c3_shards_256x256/c3-{00000..00331}.tar' \
# 	--clip_model='ViT-B/16' \
# 	--output_folder=${root}'/output/c3_emb/clip-vit-b-16' \
# 	--batch_size 4096 \
# 	--input_format='webdataset' \
# 	--output_partition_count='1'

# data: c3
# model: openclip-vit-h-14
# CUDA_VISIBLE_DEVICES=3 clip-retrieval inference.worker \
# 	--tasks='[0]' \
# 	--input_dataset=${root}'/local_data/c3_shards_256x256/c3-{00000..00331}.tar' \
# 	--clip_model='open_clip:ViT-H-14' \
# 	--output_folder=${root}'/output/c3_emb/openclip-vit-h-14' \
# 	--batch_size 1024 \
# 	--input_format='webdataset' \
# 	--output_partition_count='1'

# data: c12
# model: clip-vit-b-16
# CUDA_VISIBLE_DEVICES=0 clip-retrieval inference.worker \
# 	--tasks='[0]' \
# 	--input_dataset=${root}'/local_data/c12_shards_256x256/c12-{00000..01242}.tar' \
# 	--clip_model='ViT-B/16' \
# 	--output_folder=${root}'/output/c12_emb/clip-vit-b-16' \
# 	--batch_size 4096 \
# 	--input_format='webdataset' \
# 	--output_partition_count='1'

# data: r12
# model: clip-vit-b-16
# CUDA_VISIBLE_DEVICES=1 clip-retrieval inference.worker \
# 	--tasks='[0]' \
# 	--input_dataset=${root}'/local_data/r12_shards/{00000..01215}.tar' \
# 	--clip_model='ViT-B/16' \
# 	--output_folder=${root}'/output/r12_emb/clip-vit-b-16' \
# 	--batch_size 4096 \
# 	--input_format='webdataset' \
# 	--output_partition_count='1'

# data: y14
# model: clip-vit-b-16
# CUDA_VISIBLE_DEVICES=0 clip-retrieval inference.worker \
# 	--tasks='[0]' \
# 	--input_dataset=${root}'/local_data/y14_shards_256x256/y14-{00000..00679}.tar' \
# 	--clip_model='ViT-B/16' \
# 	--output_folder=${root}'/output/y14_emb/clip-vit-b-16' \
# 	--batch_size 4096 \
# 	--input_format='webdataset' \
# 	--output_partition_count='1' \
# 	--wds_caption_key="text"
