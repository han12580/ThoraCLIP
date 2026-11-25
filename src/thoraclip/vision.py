"""Per-second visual features from the continued-pretrained DINOv3 ViT-L/16 (Hugging Face format).

Per frame: global = [CLS ; mean patch token] (2048-d) and grid = patch tokens average-pooled to 3 x 5 (15 x 1024), both fp16.
Input: frames resized to a multiple of the patch size (bilinear, antialias), ImageNet mean / std, bf16 autocast.
"""
import numpy as np
import torch
import torch.nn.functional as F

GRID = (3, 5)


class FrameEncoder:
    def __init__(self, path, device='cuda', batch=64):
        import transformers as tf
        self.model = tf.AutoModel.from_pretrained(path).to(device).eval().requires_grad_(False)
        processor = tf.AutoImageProcessor.from_pretrained(path)
        self.mean = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(processor.image_std, device=device).view(1, 3, 1, 1)
        self.patch, self.device, self.batch = self.model.config.patch_size, device, batch

    @torch.no_grad()
    def __call__(self, frames):
        """frames: uint8 [T, H, W, 3] (or a list of such arrays of one size) -> (global [T, 2048], grid [T, 15, 1024]) fp16 numpy."""
        g_out, l_out = [], []
        for i in range(0, len(frames), self.batch):
            x = torch.from_numpy(np.ascontiguousarray(np.stack(frames[i:i + self.batch]))).to(self.device)
            x = x.permute(0, 3, 1, 2).float() / 255
            gh, gw = round(x.shape[-2] / self.patch), round(x.shape[-1] / self.patch)
            x = F.interpolate(x, (gh * self.patch, gw * self.patch), mode='bilinear', antialias=True, align_corners=False)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                hidden = self.model(pixel_values=(x - self.mean) / self.std).last_hidden_state
            patches = hidden[:, -gh * gw:].float()
            g_out.append(torch.cat([hidden[:, 0].float(), patches.mean(1)], -1).half().cpu())
            grid = patches.reshape(len(x), gh, gw, -1).permute(0, 3, 1, 2)
            l_out.append(F.adaptive_avg_pool2d(grid, GRID).flatten(2).transpose(1, 2).half().cpu())
        return torch.cat(g_out).numpy(), torch.cat(l_out).numpy()
