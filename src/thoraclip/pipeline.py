"""Inference with a released model folder.

Folder layout (written by the export tool):
  config.json                   tower / text settings, class order, instructions, logit scale
  video_tower.safetensors       temporal video tower
  text_trainable.safetensors    text projection (+ LoRA factors for the 8B model)
  queries.safetensors           class query embeddings: phase_/step_ x prototype / definition
  vision_backbone/              continued-pretrained DINOv3 ViT-L/16 (Hugging Face format)
  text_backbone/                fine-tuned Qwen3-Embedding-0.6B (absent for the 8B model, which loads the base model + LoRA)

Classification compares a clip embedding with one query embedding per class. 'prototype' queries (the paper's default) are the
mean text embedding of the training captions / descriptions of each class; 'definition' queries embed the Chinese class name and
definition (pure zero-shot). Steps are read out only among the children of the predicted phase.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from .lora import add_lora
from .text_encoder import TextEncoder
from .video_io import decode, read_images
from .video_tower import VideoTower
from .vision import FrameEncoder


def windows(n, center, width):
    """[a, b) of a `width`-second window centred on second `center`, shifted to stay inside [0, n)."""
    a = max(0, center - width // 2)
    b = min(n, a + width)
    return max(0, b - width), b


def load_text_trainable(text, state):
    """Load text_trainable.safetensors into the text tower. The file holds only the trained part (projection, plus the LoRA
    factors of the 8B model); the backbone comes from its own checkpoint. Every trained key must be present and nothing else,
    so an incomplete export fails here instead of running with a randomly initialised projection or zero LoRA."""
    result = text.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(f'unexpected text weights: {result.unexpected_keys[:5]}')
    trained = [k for k in text.state_dict() if k.startswith('projection.') or '.lora_' in k]
    missing = [k for k in trained if k not in state]
    if missing:
        raise ValueError(f'text_trainable.safetensors lacks {len(missing)} trained weights: {missing[:5]}')


class Pipeline:
    def __init__(self, folder, device='cuda', text_path=None, load_vision=True):
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoTokenizer
        torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN attention fails on very large window batches
        self.folder, self.device = Path(folder), device
        self.cfg = cfg = json.loads((self.folder / 'config.json').read_text())
        t = cfg['text']
        text_path = text_path or (str(self.folder / t['backbone']) if (self.folder / t['backbone']).exists() else t['backbone'])
        local = Path(text_path).exists()
        self.tokenizer = AutoTokenizer.from_pretrained(text_path, padding_side='right', local_files_only=local)
        hidden = AutoConfig.from_pretrained(text_path, local_files_only=local).hidden_size
        self.text = TextEncoder(SimpleNamespace(text_path=text_path, local_files_only=local, text_batch=t.get('text_batch', 32),
                                                text_feature_dim=hidden, text_gradient_checkpointing=False, dim=512))
        if t.get('lora_rank'):
            self.text.backbone.to(torch.bfloat16)
            add_lora(self.text.backbone, int(t['lora_rank']), float(t['lora_alpha']))
        load_text_trainable(self.text, load_file(self.folder / 'text_trainable.safetensors'))
        self.text = self.text.to(device).eval()
        m = cfg['tower']
        self.tower = VideoTower(use_grid=m['use_grid'], layers=m['layers'], pool=m['pool']).to(device).eval()
        self.tower.load_state_dict(load_file(self.folder / 'video_tower.safetensors'))
        self.queries = {k: v.to(device).float() for k, v in load_file(self.folder / 'queries.safetensors').items()}
        self.phase_ids, self.step_ids = cfg['phase_ids'], cfg['step_ids']
        self.step_parent = np.array(cfg['step_parent'])
        self.vision = FrameEncoder(self.folder / cfg['vision'], device) if load_vision else None

    @classmethod
    def from_pretrained(cls, folder, **kw):
        return cls(folder, **kw)

    # ---------------------------------------------------------------- features
    def encode_frames(self, frames):
        return self.vision(frames)

    def video_features(self, video, start=0., end=None):
        """1 fps features of a video file (or of [start, end) seconds of it): times from `start`, global, grid."""
        frames, times = decode(video, start, None if end is None else end - start, self.cfg['fps'], self.cfg['short_side'])
        g, grid = self.encode_frames(frames)
        return times, g, grid

    def decode_ahead(self, videos, workers=8):
        """Yield (frames, times) of each whole video file in order, decoding up to 2 x workers files ahead."""
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        it, queue = iter(videos), deque()
        dec = lambda v: decode(v, 0., None, self.cfg['fps'], self.cfg['short_side'])  # noqa: E731
        with ThreadPoolExecutor(workers) as pool:
            for v in it:
                queue.append(pool.submit(dec, v))
                if len(queue) >= 2 * workers:
                    break
            while queue:
                result = queue.popleft().result()
                v = next(it, None)
                if v is not None:
                    queue.append(pool.submit(dec, v))
                yield result

    def image_features(self, paths):
        """Features of a list of frame files (one frame per second)."""
        g, grid = zip(*(self.encode_frames(np.stack(read_images(paths[s:s + 1024], self.cfg['short_side'])))
                        for s in range(0, len(paths), 1024)))
        g, grid = np.concatenate(g), np.concatenate(grid)
        return np.arange(len(g), dtype=np.float32), g, grid

    # ---------------------------------------------------------------- embeddings
    def select(self, times, start=None, end=None):
        """Seconds of [start, end) (times relative to the features' origin), at most max_frames spread evenly; as in training."""
        if start is None:
            rows = np.arange(len(times))
        else:
            rows = np.flatnonzero((times >= start - .5) & (times < end + .5))
            if not len(rows):
                rows = np.array([int(np.argmin(np.abs(times - (start + end) / 2)))])
        mf = self.cfg['max_frames']
        if len(rows) > mf:
            rows = rows[np.linspace(0, len(rows) - 1, mf).round().astype(int)]
        return rows

    @torch.no_grad()
    def embed_segments(self, feats, segments=None, batch=256):
        """feats = (times, global, grid); segments = list of (start, end) in feature time, None = the whole range.
        Returns [len(segments), 512] normalised embeddings."""
        times, g, grid = feats
        segments = segments or [(None, None)]
        out = []
        for i in range(0, len(segments), batch):
            picks = [self.select(times, a, b) for a, b in segments[i:i + batch]]
            starts = [0. if a is None else a for a, _ in segments[i:i + batch]]
            out.append(self._tower([(times[r] - s, g[r], grid[r]) for r, s in zip(picks, starts)]))
        return torch.cat(out)

    @torch.no_grad()
    def embed_windows(self, feats, width=None, batch=512):
        """One embedding per second: the tower over a `width`-second window centred on that second (boundary-free use)."""
        _, g, grid = feats
        width = width or self.cfg['window_s']
        n, out = len(g), []
        for s in range(0, n, batch):
            idx = [windows(n, c, width) for c in range(s, min(n, s + batch))]
            out.append(self._tower([(np.arange(b - a, dtype=np.float32), g[a:b], grid[a:b]) for a, b in idx]))
        return torch.cat(out)

    def _tower(self, items):
        n = max(len(t) for t, _, _ in items)
        tb = np.zeros((len(items), n), np.float32)
        gb = np.zeros((len(items), n, 2048), np.float16)
        lb = np.zeros((len(items), n, 15, 1024), np.float16) if self.tower.use_grid else None
        mb = np.zeros((len(items), n), bool)
        for k, (t, g, l) in enumerate(items):
            tb[k, :len(t)], gb[k, :len(t)], mb[k, :len(t)] = t, g, True
            if lb is not None:
                lb[k, :len(t)] = l
        cuda = lambda a: torch.from_numpy(a).to(self.device) if a is not None else None  # noqa: E731
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return self.tower(cuda(gb), cuda(lb), cuda(tb), cuda(mb)).float()

    @torch.no_grad()
    def embed_text(self, texts, instruction=None, batch=128):
        """Caption-style texts (documents) by default; with `instruction`, texts are wrapped as Qwen queries."""
        if instruction:
            texts = [TextEncoder.format_query(t, instruction) for t in texts]
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tokenizer(texts[i:i + batch], padding=True, truncation=True, max_length=512, return_tensors='pt')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out.append(self.text(enc['input_ids'].to(self.device), enc['attention_mask'].to(self.device)))
        return torch.cat(out).float()

    # ---------------------------------------------------------------- predictions
    def phase_scores(self, emb, queries='prototype'):
        return emb @ self.queries[f'phase_{queries}'].T

    def step_scores(self, emb, phase_pred, queries='prototype'):
        """Similarity to every step; steps that are not children of the given phase are -inf, so they never rank above a
        child (similarities can be negative, so 0 would). For the paper's step mAP use metrics.routed_step_metrics, where a
        clip routed away from a step is never retrieved for it."""
        s = (emb @ self.queries[f'step_{queries}'].T).cpu().numpy()
        return np.where(self.step_parent[None] == np.asarray(phase_pred)[:, None], s, -np.inf)

    def classify(self, video, start=0., end=None, queries='prototype', top_steps=3):
        """Phase and steps of one clip ([start, end) seconds of a video file)."""
        emb = self.embed_segments(self.video_features(video, start, end))
        logits = self.phase_scores(emb, queries)[0]
        probs = torch.softmax(logits * self.cfg['logit_scale'], -1).cpu().numpy()
        p = int(probs.argmax())
        steps = self.step_scores(emb, [p], queries)[0]
        kids = [int(c) for c in np.flatnonzero(self.step_parent == p)]
        kids.sort(key=lambda c: -steps[c])
        return dict(phase_id=self.phase_ids[p], phase_probabilities={i: round(float(x), 4) for i, x in zip(self.phase_ids, probs)},
                    steps=[dict(step_id=self.step_ids[c], score=round(float(steps[c]), 4)) for c in kids[:top_steps]])

    def timeline(self, video, width=None, queries='prototype'):
        """Per-second phase for a whole video (centred windows, no boundaries needed)."""
        feats = self.video_features(video)
        scores = self.phase_scores(self.embed_windows(feats, width), queries).cpu().numpy()
        return dict(seconds=len(scores), phase_ids=[self.phase_ids[i] for i in scores.argmax(1)], scores=scores)
