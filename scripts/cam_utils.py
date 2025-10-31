import torch
import torch.nn.functional as F

@torch.no_grad()
def _normalize_cam(cam: torch.Tensor) -> torch.Tensor:
    # ReLU then min-max to [0,1]
    cam = cam.clamp(min=0)
    cam_min = cam.amin(dim=(-2, -1), keepdim=True)
    cam_max = cam.amax(dim=(-2, -1), keepdim=True)
    denom = (cam_max - cam_min).clamp(min=1e-6)
    return (cam - cam_min) / denom

def make_resnet50d_target_layers(model):
    """Return the last Bottleneck of layer4 (good CAM target for ResNet-50-D)."""
    return [model.layer4[-1]]

def gradcam_on_tensor(model, target_layer, x, target_class: int, retain_graph: bool = False):
    """
    x: (1,3,224,224) normalized tensor on device
    target_class: int
    returns: (H,W) numpy array in [0,1]
    """
    feats = []
    grads = []

    def fwd_hook(_, __, output):
        feats.append(output)

    def bwd_hook(_, grad_input, grad_output):
        grads.append(grad_output[0])

    handle_f = target_layer.register_forward_hook(fwd_hook)
    handle_b = target_layer.register_full_backward_hook(bwd_hook)

    try:
        logits = model(x)                      # (1,1000)
        score = logits[0, target_class]
        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=retain_graph)

        A = feats[-1]                          # (1,C,H,W)
        dA = grads[-1]                         # (1,C,H,W)
        weights = dA.mean(dim=(2, 3), keepdim=True)   # (1,C,1,1)
        cam = (weights * A).sum(dim=1, keepdim=True)  # (1,1,H,W)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
        cam = _normalize_cam(cam[0, 0]).detach().cpu().numpy()
        return cam
    finally:
        handle_f.remove()
        handle_b.remove()
