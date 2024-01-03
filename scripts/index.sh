
root='/home/xingyun/rewrite'

# data: c3
# model: clip-vit-b-16
# autofaiss build_index --embeddings ${root}/output/c3_emb/clip-vit-b-16/img_emb/ \
# 	--index_path ${root}/output/c3_index/clip-vit-b-16/img.index \
# 	--index_infos_path ${root}/output/c3_index/clip-vit-b-16/img.json \
# 	--save_on_disk \
# 	--max_index_memory_usage '256GB' \
# 	--current_memory_available '384GB' \
# 	--max_index_query_time_ms 1 \

# data: c12
# model: clip-vit-b-16
autofaiss build_index --embeddings ${root}/output/c12_emb/clip-vit-b-16/img_emb/ \
        --index_path ${root}/output/c12_index/clip-vit-b-16/img.index \
	--index_infos_path ${root}/output/c12_index/clip-vit-b-16/img.json \
	--save_on_disk \
	--max_index_memory_usage '256GB' \
	--current_memory_available '384GB' \
	--max_index_query_time_ms 1 \

# data: r12
# model: clip-vit-b-16
# autofaiss build_index --embeddings ${root}/output/r12_emb/clip-vit-b-16/img_emb/ \
#	--index_path ${root}/output/r12_index/clip-vit-b-16/img.index \
# 	--index_infos_path ${root}/output/r12_index/clip-vit-b-16/img.json \
# 	--save_on_disk \
# 	--max_index_memory_usage '256GB' \
# 	--current_memory_available '384GB' \
# 	--max_index_query_time_ms 1 \
# 	--min_nearest_neighbors_to_retrieve 16
