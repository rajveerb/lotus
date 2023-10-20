compress:
	@echo "[START] Compressing data..."
	@echo "Target directory: pytorch_profiles_imagenet_dataset"
	@python compressor.py --data_dir pytorch_profiles_imagenet_dataset
	@echo "[END] Compressed data"
	@echo "[START] Compressing data..."
	@echo "Target directory: pytorch_profiles_synthetic_dataset"
	@python compressor.py --data_dir pytorch_profiles_synthetic_dataset
	@echo "[END] Compressed data"

decompress:
	@echo "[START] Decompressing data..."
	@echo "Target directory: pytorch_profiles_imagenet_dataset"
	@python compressor.py --data_dir pytorch_profiles_imagenet_dataset --decompress
	@echo "[END] Decompressed data"
	@echo "[START] Decompressing data..."
	@echo "Target directory: pytorch_profiles_synthetic_dataset"
	@python compressor.py --data_dir pytorch_profiles_synthetic_dataset --decompress
	@echo "[END] Decompressed data"