# ResNet-50 Training on ImageNet-1k Using AWS EC2

Goal is to train a ResNet-50 model from scratch on the ImageNet-1k (ILSVRC 2012) dataset using PyTorch and AWS EC2 GPU instance, and the target is to achieve 75% top-1 accuracy










## Missclassified Images and Gradcam

    !python src/gradcam_viz.py \
        --num_classes 1000 \
        --checkpoint_path /content/runs/tinyimagenet_resnet50/model_best.pth.tar \
        --validation_img_path /content/tiny-imagenet-200/valid \
        --num_show 20 \
        --output_path output/gradcam_comparison.png



## Collaborators

- Jayant Guru Shrivastava (jayantgurushrivastava@gmail.com)
- Neelreddy Rebala (neelreddy.rebala@gmail.com)
- Vikas (vikasjhanitk@gmail.com)
- Divya Kamat (Divya.r.kamat@gmail.com)
