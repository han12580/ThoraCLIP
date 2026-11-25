#!/usr/bin/env python3
"""Run a released model on your own video.

  python scripts/infer.py --model weights/MODEL --video case.mp4 --start 120 --end 180     # phase + steps of one clip
  python scripts/infer.py --model weights/MODEL --video case.mp4 --timeline --out t.json   # per-second phase over the video
  python scripts/infer.py --model weights/MODEL --video case.mp4 --start 120 --end 180 \
      --text "The inferior pulmonary ligament is divided." --text "The port incisions are closed."   # video-text similarity
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from thoraclip.pipeline import Pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--video', type=Path, required=True)
    ap.add_argument('--start', type=float, default=0.)
    ap.add_argument('--end', type=float)
    ap.add_argument('--queries', default='prototype', choices=('prototype', 'definition'))
    ap.add_argument('--timeline', action='store_true', help='per-second phase over the whole video (centred windows)')
    ap.add_argument('--window', type=int, help='timeline window in seconds (default 64)')
    ap.add_argument('--text', action='append', help='caption(s) to score against the clip')
    ap.add_argument('--text-path', help='override the text backbone location')
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    model = Pipeline.from_pretrained(a.model, text_path=a.text_path)
    if a.timeline:
        t = model.timeline(a.video, a.window, a.queries)
        runs, prev = [], None
        for k, p in enumerate(t['phase_ids']):
            if p != prev:
                runs.append(dict(start_s=k, end_s=k + 1, phase_id=p))
                prev = p
            else:
                runs[-1]['end_s'] = k + 1
        out = dict(video=str(a.video), seconds=t['seconds'], segments=runs, per_second=t['phase_ids'])
    else:
        out = dict(video=str(a.video), start_s=a.start, end_s=a.end, **model.classify(a.video, a.start, a.end, a.queries))
        if a.text:
            emb = model.embed_segments(model.video_features(a.video, a.start, a.end))
            sims = (emb @ model.embed_text(a.text).T)[0].tolist()
            out['text_similarity'] = [dict(text=x, cosine=round(s, 4)) for x, s in zip(a.text, sims)]
    text = json.dumps(out, ensure_ascii=False, indent=1)
    if a.out:
        a.out.write_text(text + '\n')
    print(text if not a.timeline else json.dumps(dict(seconds=out['seconds'], segments=out['segments'][:20]), indent=1))


if __name__ == '__main__':
    main()
