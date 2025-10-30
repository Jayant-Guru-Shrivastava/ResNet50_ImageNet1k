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


## Multi-GPU Training on AWS EC2

Training large datasets like ImageNet is computationally intensive.


## Training Logs




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



