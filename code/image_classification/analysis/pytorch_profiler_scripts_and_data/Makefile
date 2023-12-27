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

augment:
	@echo "[START] Augmenting PyTorch data..."
	@echo "Target directory: pytorch_profiles_imagenet_dataset"
	@echo "Working on compact version"
	@python visualization_augmenter.py --pytorch_profiler_data_dir pytorch_profiles_imagenet_dataset --custom_log_prefix custom_log --compact
	@echo "Working on elaborate version"
	@python visualization_augmenter.py --pytorch_profiler_data_dir pytorch_profiles_imagenet_dataset --custom_log_prefix custom_log
	@echo "[END] Augmented data"
	@echo "[START] Augmenting PyTorch data..."
	@echo "Target directory: pytorch_profiles_synthetic_dataset"
	@echo "Working on compact version"
	@python visualization_augmenter.py --pytorch_profiler_data_dir pytorch_profiles_synthetic_dataset --custom_log_prefix custom_log --compact
	@echo "Working on elaborate version"
	@python visualization_augmenter.py --pytorch_profiler_data_dir pytorch_profiles_synthetic_dataset --custom_log_prefix custom_log
	@echo "[END] Augmented data"

clean_augment:
	@echo "[START] Cleaning augmented PyTorch data..."
	@echo "Target directory: pytorch_profiles_imagenet_dataset"
	@echo "Working on compact version"
	@find ./pytorch_profiles_imagenet_dataset -name "*_compact.json.xz" -type f -delete
	@echo "Working on elaborate version"
	@find ./pytorch_profiles_imagenet_dataset -name "*_elaborate.json.xz" -type f -delete
	@echo "[END] Cleaned data"
	@echo "[START] Cleaning augmented PyTorch data..."
	@echo "Target directory: pytorch_profiles_synthetic_dataset"
	@echo "Working on compact version"
	@find ./pytorch_profiles_synthetic_dataset -name "*_compact.json.xz" -type f -delete
	@echo "Working on elaborate version"
	@find ./pytorch_profiles_synthetic_dataset -name "*_elaborate.json.xz" -type f -delete
	@echo "[END] Cleaned data"
