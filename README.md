# ThoraCLIP: video–language model for lung and pleural surgery

> Weight links and the citation will be added at publication.

This repository contains three things:

- **Data processing**: rebuilds clip files and labels from the ThoraVL annotation release plus your own copies of the
  source videos.
- **Inference**: produces phase and step predictions, video and text embeddings, and a per-second phase timeline for any video.
- **Evaluation**: reproduces the paper's results for our model (doctor-verified benchmark, retrieval, boundary-free full videos,
  zero-shot Cholec80 and GraSP). Everything runs straight from the videos; no intermediate caches are needed.

## Model

| Part | What it is |
|---|---|
| Vision | DINOv3 ViT-L/16 after continued self-supervised pretraining on surgical frames. Video is sampled at 1 fps, short side 512. Each second gives a global token ([CLS; mean patch]) and a 3 × 5 patch grid. |
| Video tower | Two temporal transformer layers with an attention pool over the seconds of a clip, at most 64 seconds. Output is a 512-d embedding. |
| Text tower | Qwen3-Embedding: 0.6B fine-tuned, or 8B with LoRA r16. The last token is projected to 512-d. |
| Classes | 8 phases and 47 steps. A clip is classified by cosine similarity to one query per class. The default `prototype` queries are the mean embedding of each class's training captions. The `definition` queries embed the class name and definition instead. Steps are read out only among the children of the predicted phase. |

Released weights (&lt;links pending&gt;): `main_0p6b` and `main_8b` (sliding-window training, seed 0). Each folder contains
`config.json`, `video_tower.safetensors`, `text_trainable.safetensors`, `queries.safetensors`, `vision_backbone/` and, for
0.6B, `text_backbone/`. The 8B model loads `Qwen/Qwen3-Embedding-8B` and applies its LoRA weights on top.

## Install

```bash
pip install -r requirements.txt        # Python 3.12, CUDA GPU; ffmpeg and ffprobe on PATH
```

## Inference

```bash
# phase + steps of one clip
python scripts/infer.py --model weights/main_0p6b --video case.mp4 --start 120 --end 180
# per-second phase timeline for a whole video (64 s centred windows)
python scripts/infer.py --model weights/main_0p6b --video case.mp4 --timeline --out case_timeline.json
# video-text similarity
python scripts/infer.py --model weights/main_0p6b --video case.mp4 --start 120 --end 180 --text "The fissure is divided with a stapler."
```

```python
from thoraclip import Pipeline
model = Pipeline.from_pretrained('weights/main_0p6b')
model.classify('case.mp4', 120, 180)                        # {'phase_id': 'B004', 'phase_probabilities': {...}, 'steps': [...]}
feats = model.video_features('case.mp4', 120, 180)          # 1 fps DINOv3 features
v = model.embed_segments(feats)                             # [1, 512]
t = model.embed_text(['The fissure is divided with a stapler.'])
```

## Data processing

```bash
# RELEASE = the ThoraVL annotation folder; VIDEOS = <video_id>.mp4 files downloaded from the source pages in videos.jsonl
python scripts/prepare_data.py --release RELEASE --videos VIDEOS --out data/lung
```

The script:

- cuts every clip from its source video with the paper's settings (libx264 crf 18);
- writes the clip lists with phase / step targets, the class catalogue, and the per-video interval list;
- copies the gold benchmark.

Videos you could not obtain are listed in `data/lung/prepare_summary.json`, and their clips are skipped. We checked that the
clip rows it writes match the paper's dataset field for field.

## Evaluation (our model)

```bash
# doctor-verified benchmark (phase macro-F1, balanced accuracy, video bootstrap CI, routed step mAP),
# weak-label test set and test video<->text retrieval
python scripts/evaluate.py lung --model weights/main_0p6b --data data/lung --out results/lung.json
# boundary-free per-second phase on whole test videos; --smooth chooses smoothing on the val videos
python scripts/evaluate.py fullvideo --model weights/main_0p6b --data data/lung --smooth --out results/fullvideo.json
# zero-shot Cholec80 (test videos 49-80, SurgVLP's 1 fps frames and csv labels)
python scripts/evaluate.py cholec80 --model weights/main_0p6b --frames CHOLEC80/frames --csv CHOLEC80/csvs \
    --official-prompts SurgVLP/tests/class_prompt.txt --out results/cholec80.json
# zero-shot GraSP (test cases, 1 fps keyframes, long-term annotations)
python scripts/evaluate.py grasp --model weights/main_0p6b --grasp GraSP_1fps --out results/grasp.json
```

`lung` and `fullvideo` need `data/lung` from the data-processing step. `lung` needs only the test videos; `fullvideo --smooth`
also needs the val videos. Obtain Cholec80 (CC BY-NC-SA 4.0) and GraSP (MIT) from their providers. The paper reports the mean
over five training seeds, so a single released seed will differ slightly from the tables.

## Licence

Code: MIT (see `LICENSE`). Weights: derived from DINOv3 (DINOv3 License) and Qwen3-Embedding (Apache-2.0); the licence files
ship with the weights. ThoraVL annotations: CC BY-NC-SA 4.0 (see the dataset card).

## Citation

```
<to be added after publication>
```
