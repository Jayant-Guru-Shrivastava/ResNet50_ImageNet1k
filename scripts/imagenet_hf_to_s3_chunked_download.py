"""
ImageNet-1k Download from HuggingFace to S3
Handles parquet files with resume capability
"""

import boto3
import json
import os
import shutil
from pathlib import Path
from datasets import load_dataset
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig
import time
from datetime import datetime
import pandas as pd
from PIL import Image
import io
from huggingface_hub import HfApi
from concurrent.futures import ThreadPoolExecutor, as_completed


class ImageNetToS3Downloader:
    def __init__(self, s3_bucket, s3_prefix='imagenet-1k/', hf_token=None,
                 state_file='imagenet_download_state.json', temp_dir='/tmp/imagenet_cache',
                 max_workers=20):
        """
        Download ImageNet-1k from HuggingFace to S3 with resume capability.
        
        Args:
            s3_bucket: Your S3 bucket name
            s3_prefix: Prefix/folder in S3 (e.g., 'imagenet-1k/')
            hf_token: HuggingFace token with read access
            state_file: JSON file to track progress
            temp_dir: Temporary directory for caching
            max_workers: Number of parallel upload threads (default: 20)
        """
        # Configure S3 transfer for optimal performance
        self.transfer_config = TransferConfig(
            multipart_threshold=64 * 1024 * 1024,  # 64MB
            max_concurrency=max_workers,
            use_threads=True,
            multipart_chunksize=16 * 1024 * 1024,  # 16MB chunks
        )
        
        self.s3_client = boto3.client('s3', region_name='ap-south-1')
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix.rstrip('/') + '/'
        self.hf_token = hf_token or os.environ.get('HF_TOKEN')
        self.state_file = Path(state_file)
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.state = self._load_state()
        self.label_to_synset = None  # Will be loaded from dataset
        
        if not self.hf_token:
            raise ValueError("HuggingFace token required. Set HF_TOKEN env var or pass hf_token parameter")

    def _check_disk_space(self, required_gb=5):
        """Check available disk space and warn if insufficient."""
        stat = shutil.disk_usage(self.temp_dir)
        free_gb = stat.free / (1024**3)
        if free_gb < required_gb:
            raise RuntimeError(f"Insufficient disk space: {free_gb:.1f}GB available, {required_gb}GB required")
        print(f"  Disk space: {free_gb:.1f}GB available")
        return free_gb

    def _load_label_to_synset_mapping(self):
        """Load label to synset ID mapping from HuggingFace dataset features."""
        if self.label_to_synset is not None:
            return self.label_to_synset
        
        print("  Loading label to synset ID mapping from dataset...")
        try:
            # Try to get label names from dataset info without loading full dataset
            from datasets import get_dataset_infos
            try:
                infos = get_dataset_infos("ILSVRC/imagenet-1k")
                if 'train' in infos and 'features' in infos['train']:
                    label_feature = infos['train']['features'].get('label')
                    if label_feature and hasattr(label_feature, 'names'):
                        synset_ids = label_feature.names
                        self.label_to_synset = {i: synset_ids[i] for i in range(len(synset_ids))}
                        print(f"  ✓ Loaded mapping for {len(self.label_to_synset)} labels from dataset info")
                        return self.label_to_synset
            except Exception as e1:
                print(f"  Could not get mapping from dataset info: {e1}")
            
            # Fallback: Load dataset with streaming to get features (more efficient)
            try:
                dataset_info = load_dataset(
                    "ILSVRC/imagenet-1k",
                    split='train',
                    token=self.hf_token,
                    streaming=True,  # Use streaming to avoid loading all data
                    trust_remote_code=True
                )
                
                # Get features from streaming dataset
                if hasattr(dataset_info, 'features') and 'label' in dataset_info.features:
                    label_feature = dataset_info.features['label']
                    if hasattr(label_feature, 'names'):
                        synset_ids = label_feature.names
                        self.label_to_synset = {i: synset_ids[i] for i in range(len(synset_ids))}
                        print(f"  ✓ Loaded mapping for {len(self.label_to_synset)} labels from streaming dataset")
                        return self.label_to_synset
            except Exception as e2:
                print(f"  Could not get mapping from streaming dataset: {e2}")
            
            # Final fallback: use label index format (will work but may not match exact synset IDs)
            print(f"  Using fallback mapping (n00000000 format)")
            self.label_to_synset = {i: f"n{str(i).zfill(8)}" for i in range(1000)}
            return self.label_to_synset
            
        except Exception as e:
            print(f"  Warning: Could not load synset mapping: {e}")
            print(f"  Using fallback mapping (n00000000 format)")
            # Fallback: use label index as synset ID
            self.label_to_synset = {i: f"n{str(i).zfill(8)}" for i in range(1000)}
            return self.label_to_synset

    def _extract_images_from_parquet(self, parquet_file, output_dir, split_name):
        """
        Extract images from parquet file and save as JPEGs in ImageNet folder structure.
        
        Args:
            parquet_file: Path to parquet file
            output_dir: Directory to save JPEG images
            split_name: 'train', 'validation', or 'test'
            
        Returns:
            dict: {synset_id: [list of image paths]}
        """
        print(f"  Reading parquet file: {parquet_file.name}")
        
        # Load label to synset mapping if not already loaded
        if self.label_to_synset is None:
            self._load_label_to_synset_mapping()
        
        # Read parquet file
        df = pd.read_parquet(parquet_file)
        print(f"  Found {len(df)} images in parquet")
        
        # Create output directory structure
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        synset_images = {}
        processed_count = 0
        
        for idx, row in df.iterrows():
            try:
                # Get image and label
                image = row['image']
                label = int(row['label'])
                
                # Convert label to synset ID using proper mapping
                synset_id = self.label_to_synset.get(label, f"n{str(label).zfill(8)}")
                
                # Create synset directory
                synset_dir = output_dir / synset_id
                synset_dir.mkdir(exist_ok=True)
                
                # Save image as JPEG
                image_filename = f"{synset_id}__{idx:06d}.JPEG"
                image_path = synset_dir / image_filename
                
                # Convert PIL image to JPEG
                if hasattr(image, 'save'):
                    image.save(image_path, 'JPEG', quality=95)
                else:
                    # Handle different image formats
                    if isinstance(image, bytes):
                        img = Image.open(io.BytesIO(image))
                        img.save(image_path, 'JPEG', quality=95)
                    else:
                        # Convert to PIL Image if needed
                        img = Image.fromarray(image) if hasattr(image, 'shape') else image
                        img.save(image_path, 'JPEG', quality=95)
                
                # Track synset images
                if synset_id not in synset_images:
                    synset_images[synset_id] = []
                synset_images[synset_id].append(str(image_path))
                
                processed_count += 1
                if processed_count % 1000 == 0:
                    print(f"  Processed {processed_count}/{len(df)} images...")
                    
            except Exception as e:
                print(f"  Warning: Failed to process image {idx}: {e}")
                continue
        
        print(f"  ✓ Extracted {processed_count} images to {len(synset_images)} synset folders")
        return synset_images

    def _load_state(self):
        """Load download state from file."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'splits_completed': [],
            'parquet_files_completed': {},  # {split: [list of completed parquet files]}
            'synset_folders_uploaded': {},  # {split: {synset_id: [list of uploaded folders]}}
            'files_uploaded': {},  # {split: [list of uploaded files]} - legacy
            'started_at': None,
            'last_updated': None
        }

    def _save_state(self):
        """Save current state to file."""
        self.state['last_updated'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        print(f"State saved to {self.state_file}")

    def _file_exists_in_s3(self, s3_key):
        """Check if file already exists in S3."""
        try:
            self.s3_client.head_object(Bucket=self.s3_bucket, Key=s3_key)
            return True
        except ClientError:
            return False

    def _upload_to_s3_multipart(self, local_file, s3_key, chunk_size=100*1024*1024):
        """
        Upload file to S3 using multipart upload for reliability.
        
        Args:
            local_file: Path to local file
            s3_key: Destination S3 key
            chunk_size: Size of each part (default 100MB)
        """
        file_size = local_file.stat().st_size
        
        # For small files, use simple upload
        if file_size < chunk_size:
            print(f"  Uploading {local_file.name} ({file_size / 1024**2:.2f} MB) to S3...")
            self.s3_client.upload_file(str(local_file), self.s3_bucket, s3_key)
            return
        
        # Multipart upload for large files
        print(f"  Uploading {local_file.name} ({file_size / 1024**2:.2f} MB) using multipart...")
        mpu = self.s3_client.create_multipart_upload(
            Bucket=self.s3_bucket,
            Key=s3_key
        )
        
        parts = []
        uploaded_bytes = 0
        
        try:
            with open(local_file, 'rb') as f:
                part_num = 1
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    
                    part = self.s3_client.upload_part(
                        Bucket=self.s3_bucket,
                        Key=s3_key,
                        PartNumber=part_num,
                        UploadId=mpu['UploadId'],
                        Body=data
                    )
                    
                    parts.append({
                        'PartNumber': part_num,
                        'ETag': part['ETag']
                    })
                    
                    uploaded_bytes += len(data)
                    progress = (uploaded_bytes / file_size) * 100
                    print(f"  Upload progress: {progress:.1f}% ({uploaded_bytes / 1024**2:.1f} MB)", end='\r')
                    part_num += 1
            
            # Complete multipart upload
            self.s3_client.complete_multipart_upload(
                Bucket=self.s3_bucket,
                Key=s3_key,
                UploadId=mpu['UploadId'],
                MultipartUpload={'Parts': parts}
            )
            print(f"\n  ✓ Upload completed: s3://{self.s3_bucket}/{s3_key}")
            
        except Exception as e:
            print(f"\n  ✗ Upload failed: {e}")
            self.s3_client.abort_multipart_upload(
                Bucket=self.s3_bucket,
                Key=s3_key,
                UploadId=mpu['UploadId']
            )
            raise

    def download_split(self, split_name, max_retries=3):
        """
        Download a specific split (train/validation/test) from HuggingFace to S3 as JPEG images.
        
        Args:
            split_name: 'train', 'validation', or 'test'
            max_retries: Maximum retry attempts for failed downloads
        """
        if split_name in self.state['splits_completed']:
            print(f"\n✓ Split '{split_name}' already completed. Skipping.")
            return True
        
        print(f"\n{'='*60}")
        print(f"Processing split: {split_name} (Parquet → JPEG → S3)")
        print(f"{'='*60}")
        
        # Initialize state tracking for this split
        if split_name not in self.state['parquet_files_completed']:
            self.state['parquet_files_completed'][split_name] = []
        if split_name not in self.state['synset_folders_uploaded']:
            self.state['synset_folders_uploaded'][split_name] = {}
        
        try:
            # Check disk space before starting
            self._check_disk_space(required_gb=10)
            
            # Use streaming mode to avoid loading entire split
            print(f"Loading {split_name} split from HuggingFace (streaming mode)...")
            
            # First, get dataset info without downloading
            dataset_info = load_dataset(
                "ILSVRC/imagenet-1k",
                split=split_name,
                token=self.hf_token,
                streaming=True,  # Critical: Use streaming mode
                trust_remote_code=True
            )
            
            print(f"Dataset info:")
            print(f"  - Split: {split_name}")
            print(f"  - Features: {dataset_info.features}")
            
            # Get parquet file URLs from HuggingFace
            api = HfApi()
            
            # Get dataset repository info
            repo_info = api.repo_info("ILSVRC/imagenet-1k", repo_type="dataset")
            
            # Find parquet files for this split
            parquet_files = []
            for file_info in repo_info.siblings:
                if file_info.rfilename.endswith('.parquet') and split_name in file_info.rfilename:
                    parquet_files.append(file_info.rfilename)
            
            print(f"Found {len(parquet_files)} parquet files for {split_name} split")
            
            if not parquet_files:
                raise RuntimeError(f"No parquet files found for split: {split_name}")
            
            # Process parquet files one by one using streaming
            for idx, parquet_filename in enumerate(parquet_files, 1):
                print(f"\n[{idx}/{len(parquet_files)}] Processing: {parquet_filename}")
                
                # Download single parquet file
                parquet_path = self.temp_dir / parquet_filename
                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download the specific parquet file
                api.hf_hub_download(
                    repo_id="ILSVRC/imagenet-1k",
                    repo_type="dataset",
                    filename=parquet_filename,
                    local_dir=str(self.temp_dir),
                    token=self.hf_token
                )
                
                # Check if this parquet file is already completed
                if parquet_filename in self.state['parquet_files_completed'][split_name]:
                    print(f"[{idx}/{len(parquet_files)}] Already completed: {parquet_filename}")
                    continue
                
                # Check disk space before processing
                self._check_disk_space(required_gb=15)
                
                # Create temporary directory for JPEG extraction
                temp_jpeg_dir = self.temp_dir / f"{split_name}_jpeg_temp"
                if temp_jpeg_dir.exists():
                    shutil.rmtree(temp_jpeg_dir)
                temp_jpeg_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    # Extract images from parquet to JPEG
                    print(f"  Extracting images from {parquet_filename}...")
                    synset_images = self._extract_images_from_parquet(
                        parquet_path,
                        temp_jpeg_dir,
                        split_name
                    )
                    
                    # Upload each synset folder to S3 in parallel
                    print(f"  Uploading {len(synset_images)} synset folders to S3 (parallel uploads)...")
                    uploaded_synsets = []
                    
                    # Collect all upload tasks
                    upload_tasks = []
                    for synset_id, image_paths in synset_images.items():
                        synset_dir = temp_jpeg_dir / synset_id
                        s3_prefix = f"{self.s3_prefix}{split_name}/{synset_id}/"
                        
                        for image_path in image_paths:
                            image_name = Path(image_path).name
                            s3_key = f"{s3_prefix}{image_name}"
                            upload_tasks.append((str(image_path), s3_key, synset_id))
                    
                    # Upload all images in parallel
                    total_images = len(upload_tasks)
                    uploaded_count = 0
                    failed_count = 0
                    synset_uploaded = set()
                    synset_failed = {}
                    
                    def upload_single_image(local_path, s3_key, synset_id):
                        """Upload a single image to S3."""
                        try:
                            self.s3_client.upload_file(
                                local_path,
                                self.s3_bucket,
                                s3_key,
                                Config=self.transfer_config
                            )
                            return (True, synset_id, s3_key, None)
                        except Exception as e:
                            return (False, synset_id, s3_key, str(e))
                    
                    # Use ThreadPoolExecutor for parallel uploads
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        # Submit all upload tasks
                        future_to_task = {
                            executor.submit(upload_single_image, local_path, s3_key, synset_id): (local_path, s3_key, synset_id)
                            for local_path, s3_key, synset_id in upload_tasks
                        }
                        
                        # Process completed uploads
                        for future in as_completed(future_to_task):
                            success, synset_id, s3_key, error = future.result()
                            uploaded_count += 1
                            
                            if success:
                                synset_uploaded.add(synset_id)
                                if uploaded_count % 100 == 0:
                                    print(f"  Uploaded {uploaded_count}/{total_images} images...", end='\r')
                            else:
                                failed_count += 1
                                if synset_id not in synset_failed:
                                    synset_failed[synset_id] = 0
                                synset_failed[synset_id] += 1
                                if failed_count <= 10:  # Only print first 10 failures to avoid spam
                                    print(f"\n  Warning: Failed to upload {s3_key}: {error}")
                    
                    print(f"\n  ✓ Uploaded {uploaded_count - failed_count}/{total_images} images ({failed_count} failed)")
                    
                    # Get all unique synsets from upload tasks (for state tracking)
                    all_synsets = set(sid for _, _, sid in upload_tasks)
                    uploaded_synsets = sorted(all_synsets)
                    
                    # Print summary by synset
                    for synset_id in sorted(all_synsets):
                        total_count = sum(1 for _, _, sid in upload_tasks if sid == synset_id)
                        failed = synset_failed.get(synset_id, 0)
                        success_count = total_count - failed
                        if failed > 0:
                            print(f"  ⚠ Synset {synset_id}: {success_count}/{total_count} images uploaded ({failed} failed)")
                        else:
                            print(f"  ✓ Uploaded synset {synset_id} ({total_count} images)")
                    
                    # Update state tracking
                    self.state['parquet_files_completed'][split_name].append(parquet_filename)
                    if split_name not in self.state['synset_folders_uploaded']:
                        self.state['synset_folders_uploaded'][split_name] = {}
                    self.state['synset_folders_uploaded'][split_name][parquet_filename] = uploaded_synsets
                    self._save_state()
                    
                    print(f"  ✓ Completed parquet file: {parquet_filename}")
                    
                except Exception as e:
                    print(f"  ✗ Error processing {parquet_filename}: {e}")
                    return False
                finally:
                    # Clean up parquet file and temporary JPEG directory
                    if parquet_path.exists():
                        parquet_path.unlink()
                        print(f"  Deleted parquet file: {parquet_filename}")
                    if temp_jpeg_dir.exists():
                        shutil.rmtree(temp_jpeg_dir)
                        print(f"  Cleaned up temporary JPEG directory")
                    
                    # Check disk space after cleanup
                    self._check_disk_space(required_gb=5)
            
            # Mark split as completed
            self.state['splits_completed'].append(split_name)
            self._save_state()
            
            print(f"\n✓ Split '{split_name}' completed successfully!")
            print(f"  Total parquet files processed: {len(self.state['parquet_files_completed'][split_name])}")
            return True
            
        except Exception as e:
            print(f"\n✗ Error processing split '{split_name}': {e}")
            return False

    def download_all_splits(self, splits=['train', 'validation']):
        """
        Download all specified splits.
        
        Args:
            splits: List of splits to download (default: ['train', 'validation'])
                   Available: 'train', 'validation', 'test'
        """
        if not self.state['started_at']:
            self.state['started_at'] = datetime.now().isoformat()
            self._save_state()
        
        print(f"\n{'='*60}")
        print(f"ImageNet-1k Download to S3")
        print(f"{'='*60}")
        print(f"S3 Bucket: s3://{self.s3_bucket}/{self.s3_prefix}")
        print(f"Splits to download: {', '.join(splits)}")
        print(f"Started at: {self.state['started_at']}")
        print(f"{'='*60}\n")
        
        for split in splits:
            success = self.download_split(split)
            if not success:
                print(f"\n⚠ Download failed for split: {split}")
                print("You can resume by running the script again.")
                return False
        
        print(f"\n{'='*60}")
        print("✓ ALL DOWNLOADS COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Total splits downloaded: {len(self.state['splits_completed'])}")
        print(f"S3 Location: s3://{self.s3_bucket}/{self.s3_prefix}")
        print(f"\nTo verify files:")
        print(f"  aws s3 ls s3://{self.s3_bucket}/{self.s3_prefix} --recursive --human-readable")
        return True

    def get_progress_summary(self):
        """Print current download progress."""
        print(f"\n{'='*60}")
        print("ImageNet JPEG Download Progress Summary")
        print(f"{'='*60}")
        print(f"Started: {self.state.get('started_at', 'Not started')}")
        print(f"Last updated: {self.state.get('last_updated', 'Never')}")
        
        print(f"\nCompleted splits: {len(self.state['splits_completed'])}")
        for split in self.state['splits_completed']:
            print(f"  ✓ {split}")
        
        print(f"\nParquet files completed per split:")
        for split, files in self.state.get('parquet_files_completed', {}).items():
            print(f"  {split}: {len(files)} parquet files")
        
        print(f"\nSynset folders uploaded per split:")
        for split, synsets in self.state.get('synset_folders_uploaded', {}).items():
            total_synsets = sum(len(synset_list) for synset_list in synsets.values())
            print(f"  {split}: {total_synsets} synset folders")
        
        # Show disk space
        try:
            free_gb = self._check_disk_space(required_gb=0)
            print(f"\nCurrent disk space: {free_gb:.1f}GB available")
        except:
            print(f"\nDisk space: Unable to check")
        
        print(f"{'='*60}\n")


# Main execution script
if __name__ == "__main__":
    import sys
    
    # Configuration
    S3_BUCKET = "your-imagenet-bucket"  # CHANGE THIS
    S3_PREFIX = "imagenet-1k/"
    HF_TOKEN = "hf_xxxxxxxxxxxxx"  # CHANGE THIS or set HF_TOKEN env var
    
    # You can also set token via environment variable:
    # export HF_TOKEN=hf_xxxxxxxxxxxxx
    
    print("="*60)
    print("ImageNet-1k JPEG Downloader: HuggingFace → S3")
    print("="*60)
    
    # Validate configuration
    if S3_BUCKET == "your-imagenet-bucket":
        print("\n⚠ ERROR: Please set S3_BUCKET name in the script!")
        print("Edit the script and change 'your-imagenet-bucket' to your actual bucket name.")
        sys.exit(1)
    
    if HF_TOKEN == "hf_xxxxxxxxxxxxx" and not os.environ.get('HF_TOKEN'):
        print("\n⚠ ERROR: Please set HuggingFace token!")
        print("Either:")
        print("  1. Edit script and set HF_TOKEN variable, OR")
        print("  2. Set environment variable: export HF_TOKEN=hf_xxxxxxxxxxxxx")
        sys.exit(1)
    
    # Create downloader
    downloader = ImageNetToS3Downloader(
        s3_bucket=S3_BUCKET,
        s3_prefix=S3_PREFIX,
        hf_token=HF_TOKEN if HF_TOKEN != "hf_xxxxxxxxxxxxx" else None,
        state_file='imagenet_download_state.json',
        temp_dir='/tmp/imagenet_cache'
    )
    
    # Show current progress if resuming
    downloader.get_progress_summary()
    
    # Download all splits
    # Note: ImageNet-1k typically has 'train' and 'validation' splits
    # Add 'test' if you need it: splits=['train', 'validation', 'test']
    success = downloader.download_all_splits(splits=['train', 'validation'])
    
    if success:
        print("\n✓ JPEG conversion and upload completed! You can now use the data for PyTorch training.")
        print(f"\nNext steps:")
        print(f"1. Launch your training EC2 instance (with GPU)")
        print(f"2. Copy data from S3 to EC2 local storage:")
        print(f"   aws s3 sync s3://{S3_BUCKET}/{S3_PREFIX} /data/imagenet/")
        print(f"3. Use PyTorch ImageFolder for training:")
        print(f"   from torchvision.datasets import ImageFolder")
        print(f"   dataset = ImageFolder('/data/imagenet/train')")
        print(f"4. Start training")
    else:
        print("\n⚠ Download incomplete. Run script again to resume.")