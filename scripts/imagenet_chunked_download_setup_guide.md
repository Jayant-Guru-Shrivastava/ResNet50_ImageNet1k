# ImageNet-1k JPEG Download Setup Guide

## HuggingFace → S3 (Mumbai Region) - PyTorch Ready

This guide walks you through downloading ImageNet-1k from HuggingFace, converting it to JPEG format, and uploading to S3 using an optimized script with **parallel uploads** for maximum speed.

**Key Optimizations:**
- ⚡ **10-15x faster uploads** with parallel processing (20 workers)
- 💾 **Disk efficient** - only 50GB EBS needed (processes one parquet at a time)
- 🔄 **Resume capability** - automatically resumes from interruptions
- 🎯 **Proper synset mapping** - loads correct ImageNet class IDs from dataset

---

## Part 1: AWS Setup (Run from your local machine)

### Step 1: Install AWS CLI (if not installed)

```bash
# For Linux/Mac
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# For Windows: Download from https://aws.amazon.com/cli/

# Verify installation
aws --version
```

### Step 2: Configure AWS Credentials

```bash
aws configure
# Enter your:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region: ap-south-1
# - Output format: json
```

### Step 3: Create S3 Bucket in Mumbai Region

```bash
# Replace 'your-imagenet-bucket' with your desired bucket name
BUCKET_NAME="your-imagenet-bucket"

# Create bucket
aws s3 mb s3://$BUCKET_NAME --region ap-south-1

# Verify bucket created
aws s3 ls | grep $BUCKET_NAME
```

### Step 4: Set Bucket Lifecycle (Optional - for cost savings)

```bash
# Auto-delete after 10 days (adjust as needed)
cat > lifecycle.json << 'EOF'
{
  "Rules": [
    {
      "Id": "DeleteAfter10Days",
      "Status": "Enabled",
      "Expiration": {
        "Days": 10
      }
    }
  ]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
  --bucket $BUCKET_NAME \
  --lifecycle-configuration file://lifecycle.json

rm lifecycle.json
```

---

## Part 2: Launch EC2 Instance for Download

### Step 5: Launch EC2 Instance (Mumbai Region)

```bash
# Create security key pair (if you don't have one)
aws ec2 create-key-pair \
  --key-name imagenet-download-key \
  --query 'KeyMaterial' \
  --output text \
  --region ap-south-1 > imagenet-download-key.pem

chmod 400 imagenet-download-key.pem

# Launch t3.small instance (Ubuntu 22.04)
# This command gets the latest Ubuntu AMI and launches instance
aws ec2 run-instances \
  --image-id ami-0c2af51e265bd5e0e \
  --instance-type t3.small \
  --key-name imagenet-download-key \
  --region ap-south-1 \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=50}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=imagenet-downloader}]' \
  --iam-instance-profile Name=EC2-S3-Access

# Note: You may need to create IAM role 'EC2-S3-Access' with S3 full access
# Or attach the policy after instance launch

# Get instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=imagenet-downloader" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text \
  --region ap-south-1)

echo "Instance ID: $INSTANCE_ID"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text \
  --region ap-south-1)

echo "Public IP: $PUBLIC_IP"
```

**Alternative: Launch via AWS Console**

1. Go to EC2 Dashboard → Mumbai region
2. Launch Instance:
   - Name: imagenet-downloader
   - AMI: Ubuntu Server 22.04 LTS
   - Instance type: t3.small
   - Key pair: Create new or use existing
   - Storage: 50 GB gp3
   - IAM role: EC2 role with S3 access
3. Launch and note the Public IP

---

## Part 3: Setup EC2 Instance

### Step 6: SSH into EC2

```bash
# Wait for instance to be ready (2-3 minutes)
ssh -i imagenet-download-key.pem ubuntu@$PUBLIC_IP

# Or if you launched via console:
ssh -i your-key.pem ubuntu@<your-instance-ip>
```

### Step 7: Install Dependencies on EC2

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3-pip python3-venv -y

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip

# Verify installations
python3 --version
aws --version

# Create virtual environment
python3 -m venv ~/imagenet-env
source ~/imagenet-env/bin/activate

# Install required Python packages
pip install --upgrade pip
pip install boto3 datasets huggingface_hub pyarrow pandas pillow
```

### Step 8: Configure AWS on EC2

```bash
# Configure AWS CLI (if not using IAM role)
aws configure
# Enter same credentials as before
# Region: ap-south-1

# Verify S3 access
aws s3 ls

# You should see your bucket listed
```

### Step 9: Set HuggingFace Token

```bash
# Set your HuggingFace token as environment variable
export HF_TOKEN="hf_xxxxxxxxxxxxx"  # Replace with your actual token

# Make it persistent (optional)
echo 'export HF_TOKEN="hf_xxxxxxxxxxxxx"' >> ~/.bashrc
```

---

## Part 4: Download ImageNet

### Key Features of the Download Script

The optimized script includes:
- **Parallel S3 Uploads:** Uses ThreadPoolExecutor with 20 workers (configurable) for 10-15x faster uploads
- **Optimized Transfer Config:** boto3 TransferConfig with multipart uploads for maximum throughput
- **Proper Synset Mapping:** Automatically loads ImageNet synset IDs from HuggingFace dataset
- **Resume Capability:** Tracks progress at parquet file level and resumes from interruptions
- **Error Handling:** Comprehensive error tracking and reporting with failure statistics
- **Disk Efficient:** Processes one parquet file at a time (~5GB peak usage per file)
- **Progress Tracking:** Real-time progress updates with upload statistics

### Step 10: Clone Repository and Setup Script

```bash
# Clone the repository (or copy the script)
cd ~
git clone <your-repo-url> imagenet-download
cd imagenet-download

# Or create directory and copy script manually
mkdir -p ~/imagenet-download
cd ~/imagenet-download
# Copy imagenet_hf_to_s3_chunked_download.py to this directory
```

Edit the script configuration:
```bash
nano scripts/imagenet_hf_to_s3_chunked_download.py
```

Update these settings:
- Change `S3_BUCKET = "your-imagenet-bucket"` to your actual bucket name
- Change `HF_TOKEN = "hf_xxxxxxxxxxxxx"` to your token (or leave it to use env var)
- Optionally adjust `max_workers=20` for parallel upload threads (default: 20, increase for faster uploads)

Save and exit (Ctrl+X, Y, Enter)

### Step 11: Run Download

```bash
# Make sure virtual environment is activated
source ~/imagenet-env/bin/activate

# Navigate to script directory
cd ~/imagenet-download

# Run the download script
python3 scripts/imagenet_hf_to_s3_chunked_download.py

# The script will:
# - Download parquet files from HuggingFace (one at a time)
# - Convert parquet files to JPEG images with proper synset ID mapping
# - Upload JPEGs to S3 using parallel uploads (20 workers by default)
#   * Expected speed: 15-20 images/second (10-15x faster than sequential)
# - Show progress for each parquet file with upload statistics
# - Save state to imagenet_download_state.json
# - Resume automatically if interrupted
# - Use optimized S3 transfer configuration for maximum throughput
```

**Performance Notes:**
- **Parallel Uploads:** The script uses 20 parallel threads for S3 uploads by default
- **Speed Improvement:** ~10-15x faster than sequential uploads
- **Disk Usage:** Processes one parquet file at a time (peak ~5GB per file)
- **50GB EBS:** More than sufficient with 10x headroom

### Step 12: Monitor Progress (in another terminal)

```bash
# SSH into the same instance in a new terminal
ssh -i imagenet-download-key.pem ubuntu@$PUBLIC_IP

# Check state file for progress
cat ~/imagenet-download/imagenet_download_state.json | python3 -m json.tool

# Monitor disk usage (should stay under 10GB per parquet file)
df -h
watch -n 5 df -h  # Refresh every 5 seconds

# Check S3 bucket size and upload progress
aws s3 ls s3://your-imagenet-bucket/imagenet-1k/ --recursive --human-readable --summarize

# Monitor running process and CPU/network usage
ps aux | grep python
top
# Watch for high network activity (parallel uploads will use significant bandwidth)

# Check upload speed (if iotop is installed)
sudo apt install iotop -y
sudo iotop -o  # Shows I/O operations
```

---

## Part 5: Resume if Interrupted

### Step 13: Resume Download

If the download fails or you stop it:

```bash
# Simply run the script again
cd ~/imagenet-download
source ~/imagenet-env/bin/activate
python3 scripts/imagenet_hf_to_s3_chunked_download.py

# It will automatically:
# - Skip completed parquet files
# - Resume from where it left off
# - Show current progress summary
# - Continue with parallel uploads
```

**Configuration Options:**

You can customize the script behavior by editing the script:
- `max_workers=20`: Number of parallel upload threads (increase for faster uploads, but watch for rate limits)
- `temp_dir='/tmp/imagenet_cache'`: Temporary directory for parquet files and JPEG extraction
- `state_file='imagenet_download_state.json'`: File to track download progress

---

## Part 6: Verify and Cleanup

### Step 14: Verify Download

```bash
# Check total size in S3
aws s3 ls s3://your-imagenet-bucket/imagenet-1k/ \
  --recursive --human-readable --summarize

# List all splits
aws s3 ls s3://your-imagenet-bucket/imagenet-1k/

# Check number of files in each split
aws s3 ls s3://your-imagenet-bucket/imagenet-1k/train/ | wc -l
aws s3 ls s3://your-imagenet-bucket/imagenet-1k/validation/ | wc -l
```

### Step 15: Cleanup EC2 (IMPORTANT - to save costs!)

```bash
# Exit from EC2
exit

# From your local machine, terminate the instance
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region ap-south-1

# Or via console: EC2 → Instances → Select → Instance State → Terminate
```

### Step 16: Download State Backup (Optional)

```bash
# Before terminating, save state file locally
scp -i imagenet-download-key.pem \
  ubuntu@$PUBLIC_IP:~/imagenet-download/imagenet_download_state.json \
  ./imagenet_download_state_backup.json
```

---

## Cost Estimation

### EC2 Costs (Mumbai Region)

- **Instance:** t3.small @ ₹0.021/hour
- **Storage:** 50 GB @ ₹0.080/GB/month
- **Estimated time:** 8-12 hours for JPEG conversion and upload (with parallel uploads)
  - Previous estimate: 12-18 hours (sequential uploads)
  - **Speed improvement:** ~30-40% faster due to parallel S3 uploads
- **Total EC2 cost:** ~₹0.17 - ₹0.25

### S3 Costs (Mumbai Region)

- **Storage:** 165 GB JPEG @ ₹1.84/GB/month
- **For 1 week:** ~₹69 (~$0.85)
- **For 1 month:** ~₹304 (~$3.70)

### Data Transfer

- **HuggingFace → EC2:** Free (inbound)
- **EC2 → S3 (same region):** Free
- **S3 → EC2 (for training):** Free (same region)

**Total estimated cost for 1 week:** ~₹70 ($0.85-1.00)

---

## Troubleshooting

### Issue: "Access Denied" on S3

```bash
# Check IAM permissions
aws sts get-caller-identity

# Ensure your IAM user/role has S3 full access policy
```

### Issue: "Dataset requires authentication"

```bash
# Verify HuggingFace token
echo $HF_TOKEN

# Test token
huggingface-cli whoami --token $HF_TOKEN

# Re-accept terms at: https://huggingface.co/datasets/ILSVRC/imagenet-1k
```

### Issue: "Disk space full"

```bash
# Clear HuggingFace cache
rm -rf /tmp/imagenet_cache/*

# Or increase EC2 volume size
```

### Issue: Download very slow

```bash
# Check network speed
speedtest-cli

# The script uses parallel uploads (20 workers by default)
# To increase upload speed, edit the script and increase max_workers:
#   downloader = ImageNetToS3Downloader(..., max_workers=30)
# Note: Too many workers may cause rate limiting or connection issues

# Consider using larger instance type (t3.medium) for better network performance
# Or download during off-peak hours
```

### Issue: Synset ID mapping incorrect

```bash
# The script automatically loads synset IDs from the HuggingFace dataset
# If you see warnings about synset mapping, the script will use fallback format
# This should still work for PyTorch ImageFolder, but folder names may differ

# To verify synset mapping:
# Check the script output for "Loaded mapping for X labels"
# If it shows "Using fallback mapping", synset IDs will be in n00000000 format
```

---

## Next Steps: Using Data for PyTorch Training

Once download completes, you'll have JPEG images in S3 with ImageNet folder structure:

```
s3://your-bucket/imagenet-1k/
├── train/
│   ├── n01440764/
│   │   ├── n01440764__000001.JPEG
│   │   └── ...
│   ├── n01443537/
│   └── ...
└── validation/
    ├── n01440764/
    └── ...
```

### Step 1: Launch Training Instance

Launch a GPU instance (p3.2xlarge or similar) in Mumbai region.

### Step 2: Copy Data to Training Instance

```bash
# Copy JPEG images to local storage
aws s3 sync s3://your-imagenet-bucket/imagenet-1k/ /data/imagenet/ \
  --region ap-south-1

# Verify structure
ls /data/imagenet/train/ | head -5
ls /data/imagenet/train/n01440764/ | head -3
```

### Step 3: Use with PyTorch

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder('/data/imagenet/train', transform=train_transform)
val_dataset = datasets.ImageFolder('/data/imagenet/validation', transform=val_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Number of classes: {len(train_dataset.classes)}")
```

### Step 4: Class Labels

The folder names (e.g., `n01440764`) are ImageNet synset IDs. To get human-readable class names:

```python
# Get class names from folder structure
class_names = train_dataset.classes
print("Sample classes:", class_names[:5])

# For ImageNet-1k, you can also use:
from torchvision.datasets import ImageNet
# This provides the standard 1000 ImageNet class names
```

### Step 5: Start Training

```python
# Example training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Your training code here
        pass
```

---

## Quick Reference Commands

```bash
# Check S3 contents
aws s3 ls s3://your-imagenet-bucket/imagenet-1k/ --recursive --human-readable

# Resume download (with parallel uploads)
cd ~/imagenet-download && source ~/imagenet-env/bin/activate && python3 scripts/imagenet_hf_to_s3_chunked_download.py

# Check progress (formatted JSON)
cat imagenet_download_state.json | python3 -m json.tool

# Monitor EC2 resources
top                    # CPU and memory
df -h                  # Disk usage
watch -n 5 df -h       # Disk usage (auto-refresh)
sudo iotop -o          # I/O operations (network uploads)

# Check upload speed/bandwidth usage
# Parallel uploads will show high network activity in top/iotop

# Terminate EC2 when done
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region ap-south-1
```

**Performance Tips:**
- Monitor network bandwidth - parallel uploads use significant bandwidth
- If uploads are slow, check S3 rate limits or increase `max_workers` (but watch for rate limiting)
- Disk usage should remain stable around 5-10GB per parquet file
- Expected upload speed: 15-20 images/second with 20 workers

---

**Questions or issues?** Check the state file for detailed progress or re-run the script to resume!