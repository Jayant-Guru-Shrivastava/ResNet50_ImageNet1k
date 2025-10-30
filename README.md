# ResNet-50 Training on ImageNet-1k Using AWS EC2

Goal is to train a ResNet-50 model from scratch on the ImageNet-1k (ILSVRC 2012) dataset using PyTorch and AWS EC2 GPU instance, and the target is to achieve 75% top-1 accuracy


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









## Missclassified Images and Gradcam

    !python src/gradcam_viz.py \
        --num_classes 1000 \
        --checkpoint_path /content/runs/tinyimagenet_resnet50/model_best.pth.tar \
        --validation_img_path /content/tiny-imagenet-200/valid \
        --num_show 20 \
        --output_path output/gradcam_comparison.png


## References
- [ImageNet: A Large-Scale Hierarchical Image Database](https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf)


## Collaborators

- Jayant Guru Shrivastava (jayantgurushrivastava@gmail.com)
- Neelreddy Rebala (neelreddy.rebala@gmail.com)
- Vikas (vikasjhanitk@gmail.com)
- Divya Kamat (Divya.r.kamat@gmail.com)



