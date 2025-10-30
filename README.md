# ResNet-50 Training on ImageNet-1k Using AWS EC2

Goal is to train a ResNet-50 model from scratch on the ImageNet-1k (ILSVRC 2012) dataset using PyTorch and AWS EC2 GPU instance, and the target is to achieve 75% top-1 accuracy leveraging multi-GPU training for scalability and efficiency.

## Project Overview

This project focuses on building, training, and deploying a ResNet-50 image classification model — one of the most successful CNN architectures in computer vision history.
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
This project uses the ImageNet-1k dataset from Hugging Face Datasets:
🔗 https://huggingface.co/datasets/ILSVRC/imagenet-1k


## Training Setup and Multi-GPU Configuration

Training was performed on an AWS EC2 GPU instance (____) equipped with ____ GPUs using PyTorch with Automatic Mixed Precision (AMP) to accelerate computation and reduce memory usage.
The model was trained on the ImageNet-1k (ILSVRC 2012) dataset using the following configuration:

<img width="1264" height="337" alt="image" src="https://github.com/user-attachments/assets/58a77993-1792-4c15-9156-f87597c8eb46" />

- Model: ResNet-50D
- Epochs: 113
- Batch size: 256 (automatically split across 4 GPUs → 64 per GPU)
- Optimizer: SGD with momentum 0.9
- Learning rate schedule: Cosine annealing
- Mixed Precision: Enabled using torch.cuda.amp for faster training and reduced GPU memory footprint
- Loss Function: Cross-entropy with label smoothing (label_smoothing=0.1)

This configuration allows the model to efficiently utilize multiple GPUs on AWS for high-throughput image classification training. The use of mixed precision resulted in  faster training compared to full-precision runs, without accuracy degradation.


## Training Metrics and TensorBoard Insights
To monitor model convergence, learning rate scheduling, and performance trends, training and validation metrics are logged using TensorBoard.
Below are a few representative TensorBoard screenshots from the training process:

#### Top1 EMA accuracy
<img width="1720" height="567" alt="image" src="https://github.com/user-attachments/assets/a816fcb5-9743-4f61-af4d-e561b0f37503" />

#### Training and Validation Accuracy
<img width="1719" height="587" alt="image" src="https://github.com/user-attachments/assets/c470ffb6-71b4-4a39-b4e7-50561612dd85" />

#### Learning Rate
<img width="1472" height="444" alt="image" src="https://github.com/user-attachments/assets/ab307790-ad2c-4928-9781-c145d1195eb6" />

#### Training Loss
<img width="1464" height="385" alt="image" src="https://github.com/user-attachments/assets/0e8a031b-9cd4-412f-affe-7fd5e95753f1" />

#### Validation Loss
<img width="1457" height="380" alt="image" src="https://github.com/user-attachments/assets/821bb48d-0179-4868-8410-a7d5a60a3773" />



## Missclassified Images and Gradcam

    !python src/gradcam_viz.py \
        --num_classes 1000 \
        --checkpoint_path /content/runs/tinyimagenet_resnet50/model_best.pth.tar \
        --validation_img_path /content/tiny-imagenet-200/valid \
        --num_show 20 \
        --output_path output/gradcam_comparison.png

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


## Collaborators

- Jayant Guru Shrivastava (jayantgurushrivastava@gmail.com)
- Neelreddy Rebala (neelreddy.rebala@gmail.com)
- Vikas (vikasjhanitk@gmail.com)
- Divya Kamat (Divya.r.kamat@gmail.com)



