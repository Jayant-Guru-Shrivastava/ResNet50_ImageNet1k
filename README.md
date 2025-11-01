# ResNet-50D Training on ImageNet-1k Using AWS EC2

Goal is to train a ResNet-50D model from scratch on the ImageNet-1k (ILSVRC 2012) dataset using PyTorch and AWS EC2 GPU instance, and the target is to achieve 75% top-1 accuracy leveraging multi-GPU training for scalability and efficiency.

## Why ResNet-50D?

We used **ResNet-50D**, a modernized variant of the classic **ResNet-50**, because it incorporates several **“bag of tricks”** improvements that significantly boost ImageNet accuracy without changing model depth or parameter count drastically.

These refinements come from the paper  
🧠 *“Bag of Tricks for Image Classification with Convolutional Neural Networks”* (He et al., 2019).  
They modify the **stem and downsampling layers** to improve early feature extraction and make the model more stable during training.

### 🔍 What ResNet-50D changes compared to standard ResNet-50:
1. **Deep Stem:**  
   - Instead of a single 7×7 convolution with stride 2, it uses **three 3×3 convolutions** (stride 2 on the first).  
   - This preserves more low-level detail and reduces aliasing in early features.

2. **Downsampling tweak:**  
   - The **stride is moved from the 1×1 conv to the 3×3 conv** in the residual block.  
   - This provides smoother spatial transitions and better gradient flow.

3. **Improved pooling & BN placement:**  
   - Updated batch norm placement and average pooling lead to better feature normalization.

### 📈 Why we chose it
- **~1.5–2% higher Top-1 accuracy** on ImageNet than vanilla ResNet-50 at identical compute cost.  
- **Stable convergence** even with strong augmentations (Mixup, CutMix, RandAugment).  
- Supported directly in **timm** (`--model resnet50d`), making it easy to integrate into our reproducible training pipeline.

In short — ResNet-50D keeps the familiar ResNet architecture but uses small architectural “tricks” that give better representational power for the same training budget.

---

## Project Overview

This project focuses on building, training, and deploying a ResNet-50D image classification model — one of the most successful CNN architectures in computer vision history.
The objectives are:
- To reproduce ImageNet-1k benchmark performance from scratch on AWS GPU instances.
- To understand large-scale training pipelines, hyperparameter tuning, and convergence behavior.
- To visualize model interpretability using Grad-CAM.
- To deploy the model on Hugging Face Spaces for real-time inference.

ResNet’s key innovation — residual learning — allows training very deep networks (50+ layers) without vanishing gradients, making it ideal for large datasets like ImageNet.

## Dataset: ImageNet-1k (ILSVRC 2012)

ImageNet is one of the most significant datasets in the history of computer vision — a vast collection of over 14 million labeled images spanning 22,000 object categories, each linked to a WordNet synset (semantic category such as “dog,” “airplane,” or “apple”).
It was originally created to advance large-scale visual recognition research and became the foundation of the ImageNet Large Scale Visual Recognition Challenge (ILSVRC), which catalyzed the modern deep learning revolution.

<img width="1000" height="343" alt="image" src="https://github.com/user-attachments/assets/56c1f084-46b8-402b-b754-73495f1d651d" />


In 2012, AlexNet demonstrated a dramatic leap in accuracy on this dataset, igniting global interest in convolutional neural networks (CNNs) and GPU-based deep learning.

For this project, we use ImageNet-1k, a curated subset of ImageNet containing:
- 1,000 object categories
- ~1.28 million training images
- 50,000 validation images
- 100,000 test images (labels withheld for competition)

This subset, known as ILSVRC 2012, has become the benchmark standard for image classification, used to evaluate architectures such as VGG, ResNet, DenseNet, and Vision Transformers (ViT).

### Dataset Source
The data set is available both on [Hugging Face](https://huggingface.co/datasets/ILSVRC/imagenet-1k) as well as Kaggle  

## Why Kaggle UI download (and not Hugging Face dataset)
We downloaded ImageNet locally on macOS **via Kaggle’s browser UI** (after joining the ImageNet competition/license). We **did not** use certain community mirrors because of **format inconsistencies** that complicate standard PyTorch `ImageFolder` training, e.g.:

- Validation set not pre-arranged into class folders (requires separate mapping step).
- Class folder names/synsets or filenames not matching the official CLS‑LOC mapping.
- Occasional folder/file nesting differences that break simple scripts.

Using Kaggle UI ensures we get the official ImageNet 1K (ILSVRC2012) structure and the canonical labels.

**Local layout after unzipping (target):**
```
imagenet-1k/
└── ILSVRC
    ├── Data
    │   └── CLS-LOC
    │       ├── train/<1000 class folders>/*.JPEG
    │       └── val/<1000 class folders>/*.JPEG
    ├── Annotations/CLS-LOC/...
    └── ImageSets/CLS-LOC/...
```

---

## 1) Upload the dataset from Mac → Amazon S3

### 1.1 Create an S3 bucket
- Go to **AWS Console → S3 → Create bucket**.
- Choose a unique name, e.g. `s3://jayant-imagenet-bucket` and region (e.g., `ap-south-1`).  
- **Block public access** (recommended).

### 1.2 Create an IAM **User** for your Mac (upload only)
We’ll create a limited-access user so your Mac can upload to this one bucket.

1. **IAM → Users → Create user** → *Programmatic access*.
2. Attach a **customer managed policy** (bucket‑scoped). Example policy:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListBucket",
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::jayant-imagenet-bucket"
    },
    {
      "Sid": "PutGetObjects",
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject", "s3:DeleteObject", "s3:ListBucketMultipartUploads", "s3:AbortMultipartUpload"],
      "Resource": "arn:aws:s3:::jayant-imagenet-bucket/*"
    }
  ]
}
```
3. Save the **Access key ID** and **Secret access key** for the user.

### 1.3 Install & configure AWS CLI on Mac
```bash
# Homebrew
brew install awscli

# Configure
aws configure
# Enter the access key, secret, region (e.g., ap-south-1), output json
```

### 1.4 Upload from Mac → S3 (fast & resumable)
Tune multipart settings for large files before syncing:
```bash
aws configure set default.s3.multipart_threshold 64MB
aws configure set default.s3.multipart_chunksize 64MB
aws configure set default.s3.max_concurrent_requests 20

# Upload the dataset root
aws s3 sync ~/Downloads/imagenet-1k s3://jayant-imagenet-bucket/imagenet-1k \
  --size-only --no-progress --delete
```
> Tip: If you need to re-run uploads, `--size-only` helps skip identical files; `--delete` keeps the bucket mirrored to local.

---

## 2) Launch the GPU instance (g5.2xlarge) with S3 role and persistent EBS

### 2.1 Instance type & AMI
- **Instance**: `g5.2xlarge` (NVIDIA A10G 24GB VRAM).
- **AMI**: Prefer **Deep Learning AMI (Ubuntu 22.04)** for preinstalled NVIDIA/CUDA/PyTorch stacks.

### 2.2 EBS volume (keep data on stop/terminate)
- Root or additional EBS volume (e.g., **512–800 GB gp3** recommended).
- **Uncheck** “Delete on termination” (or set it **false**) so the volume persists if your Spot is reclaimed or instance terminated. This lets you reattach and resume quickly.

### 2.3 Spot instance
- In the **Request type**, choose **Spot** to significantly reduce cost.  
- Keep default max price; AWS uses current market price.

### 2.4 IAM **Role** for the instance (read from S3)
- Create an **IAM Role** for EC2 with **S3 read** permissions, e.g. `AmazonS3ReadOnlyAccess` or a bucket‑scoped read policy:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3ReadOnlyThisBucket",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::jayant-imagenet-bucket",
        "arn:aws:s3:::jayant-imagenet-bucket/*"
      ]
    }
  ]
}
```
- Attach this role to the instance at launch.

### 2.5 Security group
- Inbound: allow **SSH (22)** from your IP.
- (Optional) If you’ll host a service (e.g., Gradio), open appropriate ports to your IP.

---

## 3) First‑time machine setup

SSH in:
```bash
ssh -i /path/to/key.pem ubuntu@<EC2_PUBLIC_IP>
```

Base packages:
```bash
sudo apt-get update
sudo apt-get -y install git tmux htop tree unzip
# (DLAMI usually has CUDA and PyTorch pre-installed)
```

Create working dir & Python venv:
```bash
mkdir -p ~/imagenet-r50d && cd ~/imagenet-r50d
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

Install libs (if not using DLAMI preinstalls):
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
pip install timm==0.9.12 numpy pyyaml
```

Fetch the dataset from S3 onto the instance (to fast local disk, e.g., `/mnt`):
```bash
sudo mkdir -p /mnt/imagenet-1k && sudo chown ubuntu:ubuntu /mnt/imagenet-1k
aws s3 sync s3://jayant-imagenet-bucket/imagenet-1k /mnt/imagenet-1k --no-progress
tree -L 3 /mnt/imagenet-1k | head -50
```

---

## 4) Clone the training repo and prepare

```bash
cd ~/imagenet-r50d
git clone https://github.com/Jayant-Guru-Shrivastava/ResNet50_ImageNet1k .
# (If the repo has requirements.txt, do this)
# pip install -r requirements.txt
```

We trained with **PyTorch + timm** using `src/train_imagenet.py` and the `ImageFolder` layout shown above.

---

## 5) Recommended tmux layout

We use two tmux sessions:
- `train` — runs the training job
- `sync` — runs a periodic **backup to S3** of logs and checkpoints

Create them:
```bash
# Session 1: training
tmux new -s train

# In the 'train' session, run training (example schedule below)

# Session 2: background periodic backups
tmux new -s sync
# every 15 minutes, sync only logs and checkpoints
while true; do
  aws s3 sync ~/imagenet-r50d/runs/r50d-imnet s3://jayant-imagenet-bucket/checkpoints/r50d-imnet \
    --only-show-errors
  sleep 900
done
```

Detach with `Ctrl+b` then `d`. Reattach with `tmux attach -t train` (or `sync`).

---

## 6) Start training (examples)

<img width="1264" height="337" alt="image" src="https://github.com/user-attachments/assets/58a77993-1792-4c15-9156-f87597c8eb46" />

### 6.1 Fresh training run (aug on; EMA; AMP; cosine schedule)
```bash
source ~/imagenet-r50d/.venv/bin/activate
cd ~/imagenet-r50d

tmux new -s train -d 'python src/train_imagenet.py \
  --data /mnt/imagenet-1k/ILSVRC/Data/CLS-LOC \
  --model resnet50d \
  --epochs 120 \
  --batch-size 256 \
  --workers 8 \
  --aa rand-m9-mstd0.5 \
  --mixup 0.2 --cutmix 1.0 --label-smoothing 0.1 --reprob 0.25 \
  --amp --ema \
  --output runs/r50d-imnet 2>&1 | tee -a runs/r50d-imnet/console.log']

tmux attach -t train
```

### 6.2 Resume with changed augmentations (turn off Mixup/CutMix mid‑run)
From around **epoch 52** we disabled Mixup and CutMix to obtain true training accuracies. Resume like this:
```bash
LATEST=$(ls -1 runs/r50d-imnet/epoch*.pth | sort | tail -n1)

python src/train_imagenet.py \
  --data /mnt/imagenet-1k/ILSVRC/Data/CLS-LOC \
  --model resnet50d \
  --epochs 120 \
  --batch-size 256 \
  --workers 8 \
  --aa rand-m9-mstd0.5 \
  --mixup 0.0 --cutmix 0.0 --label-smoothing 0.1 --reprob 0.25 \
  --amp --ema \
  --output runs/r50d-imnet \
  --resume "$LATEST"
```

### 6.3 Turn off Random Erasing later in training
From around **epoch 83** we set **`--reprob 0.0`** (random erasing off) to improve late‑stage stability:
```bash
LATEST=$(ls -1 runs/r50d-imnet/epoch*.pth | sort | tail -n1)

python src/train_imagenet.py \
  --data /mnt/imagenet-1k/ILSVRC/Data/CLS-LOC \
  --model resnet50d \
  --epochs 120 \
  --batch-size 256 \
  --workers 8 \
  --aa rand-m9-mstd0.5 \
  --mixup 0.0 --cutmix 0.0 --label-smoothing 0.1 --reprob 0.0 \
  --amp --ema \
  --output runs/r50d-imnet \
  --resume "$LATEST"
```

> **Note on logging**: At the start, with Mixup/CutMix ON, we **did not log** `train_top1`/`train_top5` (they are not meaningful under soft targets). After disabling Mixup/CutMix (~epoch **52**), we **started logging** training accuracies from ~epoch **63**.

---

## 7) Accuracy milestones (from `runs/r50d-imnet/log.jsonl`)

From our recorded run:
- **EMA Top‑1 ≥ 75%** at **epoch ~80** (`ema_top1: 75.082` @ epoch 80).
- **EMA Top‑1 ≥ 78%** at **epoch ~101** (`ema_top1: 78.11` @ epoch 101).
- **Val Top‑1 ≥ 75%** at **epoch ~96** (e.g., `val_top1: 75.144` @ epoch 95).
- Continued improvement through epochs ~113 with `val_top1 ≈ 78.502` and `ema_top1 ≈ 78.474`.

You can compute slopes quickly to judge if training is still improving:
```bash
python - <<'PY'
import json, numpy as np, pathlib
p=pathlib.Path("runs/r50d-imnet/log.jsonl")
rows=[json.loads(x) for x in p.read_text().splitlines()]
def slope(key, k=10):
    ys=[r.get(key) for r in rows if r.get(key) is not None][-k:]
    xs=np.arange(len(ys))
    m=np.polyfit(xs, ys, 1)[0]
    return m, ys
for key in ["ema_top1","val_top1","train_top1"]:
    try:
        m, ys = slope(key, 10)
        print(f"{key}: last10={ys}  slope={m:.3f} pts/epoch")
    except Exception as e:
        pass
PY
```

---

## Training Metrics and TensorBoard Insights
To monitor model convergence, learning rate scheduling, and performance trends, training and validation metrics are logged using TensorBoard.
Below are a few representative TensorBoard screenshots from the training process:

#### Top1 EMA accuracy
<img width="1720" height="567" alt="image" src="https://github.com/user-attachments/assets/a816fcb5-9743-4f61-af4d-e561b0f37503" />

#### Training and Validation Accuracy
<img width="1719" height="587" alt="image" src="https://github.com/user-attachments/assets/c470ffb6-71b4-4a39-b4e7-50561612dd85" />

#### Learning Rate
<img width="1472" height="444" alt="image" src="https://github.com/user-attachments/assets/ab307790-ad2c-4928-9781-c145d1195eb6" />

#### Training and Validation Loss
<img width="840" height="401" alt="image" src="https://github.com/user-attachments/assets/71d30145-7e8b-43ef-b426-a619c6db0a6c" />

# 🔍 Misclassification Analysis and Grad-CAM Visualization

To better understand where the network struggles, we conducted a **post-training error analysis** on the ImageNet-1k validation set.

After achieving **EMA Top-1 ≈ 78%**, we analyzed the **misclassified images** using our custom scripts:
- `scripts/eval_dump_miscls.py` — dumps model predictions and confidence scores for each validation image.  
- `scripts/plot_miscls.py` — visualizes misclassified examples, with and without Grad-CAM overlays.  
- `scripts/aggregate_errors.py` — summarizes top confusions and per-class accuracy.

## ⚙️ Command used
```bash
python scripts/eval_dump_miscls.py   --data /mnt/imagenet-1k/ILSVRC/Data/CLS-LOC   --weights runs/r50d-imnet/best_model.pth   --model resnet50d   --batch-size 32   --workers 4   --device mps   --out runs/analysis/miscls.csv

python scripts/plot_miscls.py   --csv runs/analysis/miscls.csv   --weights runs/r50d-imnet/best_model.pth   --arch resnet50d   --num 32   --device mps   --out_no_cam runs/analysis/misclassified_report_no_gradcam.png   --out_cam     runs/analysis/misclassified_report.png
```

## 🧠 What the analysis shows
- **True vs Predicted labels** are displayed for each misclassified image, along with the model’s top-1 confidence.  
- **Grad-CAM overlays** highlight *where* the model focused when making the wrong prediction, revealing attention biases or dataset ambiguities.  
- Some classes (e.g., similar dog breeds or objects with overlapping shapes) remain challenging even for human annotators.  

## 📊 Sample Results
| Without Grad-CAM | With Grad-CAM |
|------------------|---------------|
| <img src="runs/analysis/misclassified_report_no_gradcam.png" width="1000"> | <img src="runs/analysis/misclassified_report.png" width="1000"> |

> 🔴 **Red areas** = regions with highest activation;  
> 🔵 **Blue areas** = least influential regions.  

This interpretability study demonstrates how **ResNet-50D’s attention** correlates with visual features and where it occasionally fails — for example:
- confusing **“ping-pong ball” vs “beaker”** (similar texture and context),
- **dog breeds** with minor facial differences,
- **tools vs utensils** with overlapping shapes.

## 💡 Insights
- High-confidence misclassifications (> 95%) often arise from **fine-grained categories**.  
- Low-confidence misclassifications (< 30%) typically indicate **ambiguous visual cues or occlusions**.  
- Grad-CAM maps validate that the network is largely focusing on semantically relevant regions, confirming that **training and representation learning were successful**.


## Model Deployment on HuggingFace

The trained ResNet-50 model is deployed as an interactive demo on Hugging Face Spaces, making it easy to test image classification directly in the browser without any setup.

🔗 [Live Demo: ResNet-50 ImageNet Grad-CAM on Hugging Face](https://huggingface.co/spaces/jayantgurushrivastava/resnet50d-imagenet-gradcam)

The deployment was built using Gradio, which provides an intuitive web interface for inference.
Users can upload an image, and the app will:
- Classify the image into one of the 1,000 ImageNet categories.
- Display the top-5 predicted classes with confidence scores.
- Generate a Grad-CAM heatmap overlay highlighting the key regions that influenced the model’s decision.

<img width="1756" height="961" alt="image" src="https://github.com/user-attachments/assets/316230d7-617d-4918-9414-cb57430f2c9d" />


## References
- [ImageNet: A Large-Scale Hierarchical Image Database](https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf)
- [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187)

## Collaborators

- Jayant Guru Shrivastava (jayantgurushrivastava@gmail.com)
- Neelreddy Rebala (neelreddy.rebala@gmail.com)
- Vikas (vikasjhanitk@gmail.com)
- Divya Kamat (Divya.r.kamat@gmail.com)



