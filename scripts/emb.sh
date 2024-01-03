clip-retrieval inference.worker \
--tasks="[0]" \
--input_dataset="local_data/c3_shards/{00000..00331}.tar" \
--output_folder="" \
--batch_size=2048 \
--input_format="webdataset" \
--output_partition_count="1" \
--clip_model="facebook/dino-vitb16" \
--enable_text="False"
