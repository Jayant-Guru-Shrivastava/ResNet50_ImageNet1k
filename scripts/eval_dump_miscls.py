import argparse, json, os, csv, sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        path = self.samples[index][0]
        return img, target, path

def read_synset_map(cls_loc_dir: str):
    import os
    candidates = [
        os.path.join(cls_loc_dir, "../../../LOC_synset_mapping.txt"),
        os.path.join(cls_loc_dir, "../../LOC_synset_mapping.txt"),
        os.path.join(cls_loc_dir, "../LOC_synset_mapping.txt"),
        os.path.join(cls_loc_dir, "LOC_synset_mapping.txt"),
    ]
    mapping_path = next((p for p in candidates if os.path.isfile(p)), None)
    syn2name = {}
    if mapping_path:
        with open(mapping_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    syn, name = parts[0], parts[1].strip()
                    syn2name[syn] = name
    return syn2name

def strip_module(state):
    return {k.replace("module.", ""): v for k, v in state.items()}

def select_state_dict(ckpt):
    """
    Try hard to find the actual weights in common layouts.
    Return a flat state_dict or raise.
    """
    if isinstance(ckpt, dict):
        # obvious keys first
        for k in ["state_dict_ema", "model_ema", "ema", "ema_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return strip_module(ckpt[k])
        for k in ["state_dict", "model", "net"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return strip_module(ckpt[k])
        # flat?
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return strip_module(ckpt)
    # nothing matched
    raise RuntimeError(
        "Could not locate a state_dict in checkpoint. "
        "Looked for keys: state_dict_ema/model_ema/ema/ema_state_dict/state_dict/model/net."
    )

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Path to .../ILSVRC/Data/CLS-LOC')
    ap.add_argument('--weights', required=True, help='Path to .pth checkpoint')
    ap.add_argument('--model', default='resnet50d')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    ap.add_argument('--out', default='runs/analysis/miscls.csv')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    tfm = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])

    valdir = os.path.join(args.data, 'val')
    dataset = ImageFolderWithPaths(valdir, tfm)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=True, shuffle=False)

    device = torch.device(args.device)
    model = timm.create_model(args.model, pretrained=False, num_classes=1000)

    ckpt = torch.load(args.weights, map_location='cpu')
    state = select_state_dict(ckpt)  # <- robust pickup
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded weights. missing={len(missing)} unexpected={len(unexpected)}")
    if missing: print("  sample missing:", missing[:8])
    if unexpected: print("  sample unexpected:", unexpected[:8])

    model.to(device).eval()

    idx_to_syn = {i: syn for syn, i in dataset.class_to_idx.items()}
    syn_to_name = read_synset_map(args.data)
    def pretty(idx):
        syn = idx_to_syn[idx]
        return syn, syn_to_name.get(syn, syn)

    # QUICK SANITY on first few batches: average top1 prob should NOT be ~0.001
    probe_total, probe_sum = 0, 0.0

    fields = ['img_path','true_idx','true_synset','true_label','pred_idx','pred_synset','pred_label','prob_top1','topk_indices','topk_probs']
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(fields)

        for batch_idx, batch in enumerate(tqdm(loader, total=len(loader))):
            imgs, targets, paths = batch
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            p_topk, i_topk = probs.topk(args.topk, dim=1)
            preds = i_topk[:, 0]

            # collect probe stats on first 2 batches
            if batch_idx < 2:
                probe_total += imgs.size(0)
                probe_sum += p_topk[:,0].sum().item()

            for b in range(imgs.size(0)):
                ti = int(targets[b].item())
                pi = int(preds[b].item())
                tsyn, tname = pretty(ti)
                psyn, pname = pretty(pi)
                w.writerow([
                    paths[b], ti, tsyn, tname, pi, psyn, pname,
                    float(p_topk[b,0].item()),
                    json.dumps([int(x) for x in i_topk[b].tolist()]),
                    json.dumps([float(x) for x in p_topk[b].tolist()])
                ])

    if probe_total:
        avg_top1 = probe_sum / probe_total
        print(f"[sanity] avg top1 prob on first ~{probe_total} imgs: {avg_top1:.3f}")
        if avg_top1 < 0.01:
            print("WARNING: average top1 prob ~0 → model likely not loaded (uniform softmax). "
                  "Double-check that the checkpoint head is 1000 classes and we picked the right key.", file=sys.stderr)

    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
