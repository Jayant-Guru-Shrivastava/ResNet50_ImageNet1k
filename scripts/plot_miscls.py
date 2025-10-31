import argparse, random, os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch, timm
import torchvision.transforms as T

from cam_utils import make_resnet50d_target_layers, gradcam_on_tensor

def load_df(csv_path):
    df = pd.read_csv(csv_path)
    df['is_miscls'] = df['true_idx'] != df['pred_idx']
    return df

def load_model(weights, arch='resnet50d', device='cpu'):
    model = timm.create_model(arch, pretrained=False, num_classes=1000)
    state = torch.load(weights, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model

def build_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def build_vis_transform():
    return T.Compose([T.Resize(256), T.CenterCrop(224)])

def overlay_cam(rgb_img_float, cam_0to1, alpha=0.35):
    # rgb_img_float: HxWx3 in [0,1], cam_0to1: HxW in [0,1]
    import matplotlib.cm as cm
    heatmap = cm.get_cmap('jet')(cam_0to1)[..., :3]   # HxWx3
    return (alpha * heatmap + (1 - alpha) * rgb_img_float).clip(0, 1)

def make_grid(paths, titles, pred_indices, out_png, with_cam=False, model=None, cam_target_layers=None, device='cpu'):
    cols = 8
    rows = int(np.ceil(len(paths) / cols))

    # 🧠 Increase figure size & resolution slightly
    plt.figure(figsize=(cols * 3.2, rows * 3.6), dpi=220)

    tf = build_transform()
    vis_tf = build_vis_transform()
    target_layer = cam_target_layers[0] if (with_cam and cam_target_layers) else None

    for i, p in enumerate(paths):
        ax = plt.subplot(rows, cols, i + 1)
        img = Image.open(p).convert('RGB')
        vis = np.asarray(vis_tf(img)).astype(np.float32) / 255.0

        if with_cam:
            x = tf(img).unsqueeze(0).to(device)
            pred_idx = int(pred_indices[i])

            cam = gradcam_on_tensor(model, target_layer, x, pred_idx)
            if np.max(cam) <= 1e-6:
                with torch.no_grad():
                    logits = model(x)
                alt_idx = int(torch.argmax(logits, dim=1).item())
                cam = gradcam_on_tensor(model, target_layer, x, alt_idx)

            overlay = (overlay_cam(vis, cam) * 255.0).astype(np.uint8)
            ax.imshow(overlay)
        else:
            ax.imshow((vis * 255.0).astype(np.uint8))

        # 🧩 smaller title font + more vertical spacing
        ax.set_title(titles[i], fontsize=8, pad=8)
        ax.axis('off')

    # 🔧 adjust spacing between rows/columns
    plt.subplots_adjust(
        left=0.02, right=0.98, top=0.97, bottom=0.03,
        wspace=0.25, hspace=0.45  # <- more horizontal & vertical space
    )

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png)
    plt.close()
    print(f"Saved {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--arch', default='resnet50d')
    ap.add_argument('--num', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_no_cam', default='runs/analysis/misclassified_report_no_gradcam.png')
    ap.add_argument('--out_cam', default='runs/analysis/misclassified_report.png')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_cam), exist_ok=True)
    random.seed(args.seed)

    df = load_df(args.csv)
    mis = df[df.is_miscls].copy()
    if len(mis) == 0:
        print("No misclassifications in CSV—check inputs.")
        return

    # Most confident mistakes first (display as percentage with one decimal)
    mis.sort_values('prob_top1', ascending=False, inplace=True)
    sample = mis.sample(n=args.num, random_state=args.seed)


    # Build human-friendly titles (no numeric class id)
    def fmt_prob(p):
        try:
            return f"{float(p)*100:.1f}%"
        except Exception:
            return str(p)

    titles = [
        f"True: {row['true_label']}\nPred: {row['pred_label']} ({fmt_prob(row['prob_top1'])})"
        for _, row in sample.iterrows()
    ]
    paths = sample['img_path'].tolist()
    pred_indices = [int(x) for x in sample['pred_idx'].tolist()]

    # Plain grid
    make_grid(paths, titles, pred_indices, args.out_no_cam, with_cam=False)

    # Grad-CAM grid
    device = torch.device(args.device)
    model = load_model(args.weights, args.arch, device)
    cam_layers = make_resnet50d_target_layers(model)
    make_grid(paths, titles, pred_indices, args.out_cam, with_cam=True, model=model, cam_target_layers=cam_layers, device=device)

if __name__ == '__main__':
    main()
