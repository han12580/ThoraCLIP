#!/usr/bin/env python3
"""Evaluate a released model end to end: decode the videos, extract features, score; nothing is cached between runs.

  lung       clip level on the prepared dataset (scripts/prepare_data.py): doctor-verified gold benchmark (phase macro-F1,
             balanced accuracy, video-level bootstrap CI, routed step mAP), the weak-label test set, and test video<->text
             retrieval (R@1/5/10).
  fullvideo  boundary-free per-second phase on whole test videos (centred windows), scored on labelled seconds; with
             --smooth the smoothing is chosen on the val videos (pooled macro-F1) and applied to test.
  cholec80   zero-shot phase recognition on Cholec80 test videos 49-80 (1 fps frames and csv labels as prepared by SurgVLP).
  grasp      zero-shot phase and step recognition on the GraSP test set (1 fps keyframes, long-term annotations).

Examples
  python scripts/evaluate.py lung --model weights/MODEL --data data/lung --out results/lung.json
  python scripts/evaluate.py fullvideo --model weights/MODEL --data data/lung --smooth --out results/fullvideo.json
  python scripts/evaluate.py cholec80 --model weights/MODEL --frames CHOLEC/frames --csv CHOLEC/csvs \
      --official-prompts SurgVLP/tests/class_prompt.txt --out results/cholec80.json
  python scripts/evaluate.py grasp --model weights/MODEL --grasp GraSP_1fps --out results/grasp.json
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from thoraclip import metrics as M  # noqa: E402
from thoraclip.pipeline import Pipeline  # noqa: E402

CHOLEC_PHASES = ['Preparation', 'Calot triangle dissection', 'Clipping and cutting', 'Gallbladder dissection',
                 'Gallbladder packaging', 'Cleaning and coagulation', 'Gallbladder retraction']
CHOLEC_INSTRUCTION = 'Given a description of a laparoscopic cholecystectomy video clip, retrieve the surgical phase it shows'
GRASP_INSTRUCTION = {'phase': 'Given a description of a robot-assisted radical prostatectomy video clip, retrieve the surgical phase it shows',
                     'step': 'Given a description of a robot-assisted radical prostatectomy video clip, retrieve the surgical step it shows'}


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def log(**kw):
    print(json.dumps(kw, ensure_ascii=False), flush=True)


def has_text(r):
    return any(isinstance(r.get(k), str) and r[k].strip() for k in ('caption', 'description'))


def eligible(rows):
    return [r for r in rows if r.get('phase_training_eligible') and r.get('phase_target') is not None and has_text(r)]


# Labels are mapped by class ID into the model's own order (config.json phase_ids / step_ids), never taken from the integer
# phase_target / step_targets of the prepared rows: those follow the dataset's taxonomy.json, and a different taxonomy order
# would still score, just against the wrong columns.
def phase_label(model, phase_id):
    if phase_id not in model.phase_ids:
        raise ValueError(f'phase {phase_id} is not in the model classes')
    return model.phase_ids.index(phase_id)


def step_labels(model, step_ids):
    unknown = set(step_ids) - set(model.step_ids)
    if unknown:
        raise ValueError(f'steps {sorted(unknown)} are not in the model classes')
    return [int(s in step_ids) for s in model.step_ids]


# ================================================================= lung clip level
def lung(args, model):
    data = args.data
    trained = model.cfg['trained_steps']
    test = eligible(jsonl(data / 'in_scope/test.jsonl'))
    gold_rows = jsonl(data / 'gold/gold_segments.jsonl')
    if args.limit_videos:  # smoke test: the first N test videos that have gold segments
        keep = list(dict.fromkeys(s['video_id'] for s in gold_rows))[:args.limit_videos]
        test = [r for r in test if r['video_id'] in keep]
    by_clip = {r['clip_id']: r for r in test}
    feats = {}
    t0 = time.time()
    # one decode + DINOv3 pass per test clip; clips are decoded ahead in threads while the GPU encodes
    for k, (r, (frames, times)) in enumerate(zip(test, model.decode_ahead([data / 'in_scope' / r['video'] for r in test], args.workers))):
        feats[r['clip_id']] = (times, *model.encode_frames(frames))
        if (k + 1) % 100 == 0:
            log(clips=k + 1, of=len(test), minutes=round((time.time() - t0) / 60, 1))

    def embed(items):
        out = []
        for it in items:
            seg = None if it['whole'] else [(it['start'], it['end'])]
            out.append(model.embed_segments(feats[it['clip_id']], seg))
        return torch.cat(out)

    # items: weak test clips (whole) and gold segments (whole clip or the doctor's sub-range, cut from the clip's seconds)
    weak = [dict(clip_id=r['clip_id'], whole=True, video_id=r['video_id'], phase=phase_label(model, r['phase_id']),
                 steps=step_labels(model, r['step_ids']) if r.get('step_training_eligible') else None) for r in test]
    gold = []
    for s in gold_rows:
        r = by_clip.get(s['clip_id'])
        if r is None:  # the clip has no caption / description: not scored (1 segment)
            continue
        if not r['start_s'] - 1e-3 <= s['start_s'] < s['end_s'] <= r['end_s'] + 1e-3:
            raise ValueError(f"gold segment outside its clip: {s['segment_id']}")
        whole = abs(s['start_s'] - r['start_s']) < 1e-3 and abs(s['end_s'] - r['end_s']) < 1e-3
        start = 0. if whole else s['start_s'] - r['start_s']
        gold.append(dict(clip_id=r['clip_id'], whole=whole, start=start, end=start + (r['duration_s'] if whole else s['end_s'] - s['start_s']),
                         video_id=s['video_id'], segment_id=s['segment_id'], phase=phase_label(model, s['phase_id']),
                         steps=step_labels(model, s['step_ids'])))

    def zero_shot(items, emb, q):
        pred = model.phase_scores(emb, q).argmax(1).cpu().numpy()
        y = np.array([i['phase'] for i in items])
        videos = np.array([i['video_id'] for i in items])
        out = dict(n=len(items), videos=len(set(videos)), phase=M.macro_scores(y, pred), phase_bootstrap=M.video_bootstrap(y, pred, videos))
        keep = np.array([i['steps'] is not None for i in items])
        sims = (emb @ model.queries[f'step_{q}'].T).cpu().numpy()[keep]
        ys = np.array([i['steps'] for i in items if i['steps'] is not None])
        out['steps'] = M.routed_step_metrics(ys, sims, pred[keep], y[keep], model.step_parent, videos[keep], trained, model.step_ids)
        return out

    e_weak, e_gold = embed(weak), embed(gold)
    result = {q: dict(gold=zero_shot(gold, e_gold, q), weak_test=zero_shot(weak, e_weak, q)) for q in ('prototype', 'definition')}

    # retrieval on the weak test clips: gallery = each clip's caption (first text)
    texts = [r.get('caption') or r.get('description') for r in test]
    t_emb = model.embed_text(texts)
    # spans on the SOURCE-video timeline, so only clips of the same video that really overlap by > 50% are not negatives
    # (clip-relative spans starting at 0 made every pair of clips from one video look overlapping)
    rows = [dict(source_id='lung:' + r['video_id'], clip_uid=r['clip_id'], decode_start_s=float(r['start_s']),
                 decode_end_s=float(r['start_s']) + float(r['duration_s']), alignment_text=x) for r, x in zip(test, texts)]
    positive, allowed = M.alignment_masks(rows, rows)
    scores = (e_weak @ t_emb.T).float().cpu()
    retrieval = dict(v2t=M.retrieval_direction(scores, positive, allowed), t2v=M.retrieval_direction(scores.T, positive.T, allowed.T))
    summary = dict(gold_phase_f1=result['prototype']['gold']['phase']['macro_f1'],
                   gold_phase_bacc=result['prototype']['gold']['phase']['balanced_accuracy'],
                   gold_phase_f1_95ci=result['prototype']['gold']['phase_bootstrap']['macro_f1_95ci'],
                   gold_step_mAP=result['prototype']['gold']['steps']['predicted_phase']['mAP'],
                   gold_step_mAP_frequent=result['prototype']['gold']['steps']['predicted_phase']['mAP_frequent'],
                   definition_gold_phase_f1=result['definition']['gold']['phase']['macro_f1'],
                   test_phase_f1=result['prototype']['weak_test']['phase']['macro_f1'],
                   # None (no query with a true negative) stays None -> JSON null, never 0
                   **{f'test_{d}_R{k}': None if retrieval[d][f'R{k}'] is None else 100 * retrieval[d][f'R{k}']
                      for d in ('v2t', 't2v') for k in (1, 5, 10)})
    return dict(summary=summary, queries=result, retrieval=retrieval, gold_segments_scored=len(gold), test_clips=len(test))


# ================================================================= full videos, boundary-free
def second_labels(intervals, n):
    """label per second k; -1 = unlabelled, or covered by two clips with different phases."""
    lab, seen = np.full(n, -1), defaultdict(set)
    for a, b, y in intervals:
        for k in range(max(0, int(np.ceil(a))), min(n, int(np.ceil(b)))):
            seen[k].add(y)
    for k, ys in seen.items():
        if len(ys) == 1:
            lab[k] = next(iter(ys))
    return lab


def split_videos(data, split, model):
    rows = {r['clip_id']: r for r in eligible(jsonl(data / f'in_scope/{split}.jsonl'))}
    out = []
    for v in jsonl(data / 'videos.jsonl'):
        ivs = [(float(i['start_s']), float(i['end_s']), phase_label(model, rows[i['clip_id']]['phase_id'])) for i in v['intervals'] if i['clip_id'] in rows]
        if ivs:
            out.append(dict(video_id=v['video_id'], path=data / v['video'], intervals=ivs))
    return out


def fullvideo(args, model):
    def run(split):
        out = []
        for k, v in enumerate(split_videos(args.data, split, model)[:args.limit_videos or None]):
            feats = model.video_features(v['path'])
            s = model.phase_scores(model.embed_windows(feats, args.window), args.query).cpu().numpy()
            out.append(dict(video=v['video_id'], scores=s, labels=second_labels(v['intervals'], len(s))))
            log(split=split, video=k + 1, seconds=len(s))
        return out

    def evaluate(vs, method, value, sigma, bootstrap=False):
        from sklearn.metrics import f1_score
        ys, ps, vid, accs, f1s = [], [], [], [], []
        for v in vs:
            p = M.smooth_predict(v['scores'], method, value, sigma)
            keep = v['labels'] >= 0
            if not keep.any():
                continue
            y, q = v['labels'][keep], p[keep]
            ys.append(y), ps.append(q), vid.append(np.array([v['video']] * len(y)))
            accs.append(float((y == q).mean())), f1s.append(float(f1_score(y, q, average='macro')))
        y, p = np.concatenate(ys), np.concatenate(ps)
        out = dict(per_video_acc=float(np.mean(accs)), per_video_f1=float(np.mean(f1s)), pooled_f1=M.macro_scores(y, p)['macro_f1'],
                   scored_seconds=int(len(y)))
        if bootstrap:
            out['pooled_bootstrap'] = M.video_bootstrap(y, p, np.concatenate(vid))
        return out

    test = run('test')
    result = dict(window_s=args.window or model.cfg['window_s'], query=args.query, unsmoothed=evaluate(test, 'none', 0, 0., True))
    if args.smooth:
        val = run('val')
        sigma = float(np.mean([v['scores'].std(1).mean() for v in val]))
        table = [dict(method=m, value=x, val_pooled_f1=evaluate(val, m, x, sigma)['pooled_f1']) for m, x in M.SMOOTHING_GRID]
        best = max(table, key=lambda r: r['val_pooled_f1'])
        result.update(sigma=sigma, val_table=table, chosen=dict(method=best['method'], value=best['value']),
                      smoothed=evaluate(test, best['method'], best['value'], sigma, True))
    return result


# ================================================================= Cholec80
def cholec80(args, model):
    import os
    index = {}
    for dirpath, _, files in os.walk(args.frames):
        for f in files:
            if f.endswith(('.png', '.jpg')):
                index[f] = os.path.join(dirpath, f)
    a, b = map(int, args.videos.split('-'))
    prompts = dict(names=CHOLEC_PHASES)
    if args.official_prompts:
        prompts['official'] = [line.strip() for line in Path(args.official_prompts).read_text().splitlines() if line.strip()]
        assert len(prompts['official']) == 7
    queries = {k: model.embed_text(ps, CHOLEC_INSTRUCTION) for k, ps in prompts.items()}
    out = {k: [] for k in prompts}
    for v in list(range(a, b + 1))[:args.limit_videos or None]:
        rows = [line.split(',') for line in (args.csv / f'video{v:02d}.csv').read_text().splitlines()[1:] if line.strip()]
        rows = rows[:args.limit_frames or None]
        # SurgVLP loader: csv frame k is the file <video>_<k+1>.png; csv labels are 1..7
        names = ['{}_{}.png'.format(Path(p).stem.split('_')[0], int(Path(p).stem.split('_')[1]) + 1) for p, _ in rows]
        missing = [n for n in names if n not in index]
        if missing:
            raise FileNotFoundError(f'{len(missing)} Cholec80 frames of video{v:02d} not under --frames (first: {missing[0]})')
        paths = [index[n] for n in names]
        labels = np.array([int(y) - 1 for _, y in rows])
        emb = model.embed_windows(model.image_features(paths), args.window)
        for k, q in queries.items():
            out[k].append(dict(video=v, labels=labels[:len(emb)], pred=(emb @ q.T).argmax(1).cpu().numpy()))
        log(video=v, frames=len(paths))
    return dict(window_s=args.window or model.cfg['window_s'], prompts=prompts,
                protocol='SurgVLP zero-shot: per-video accuracy and macro-F1, mean over test videos, 1 fps frames',
                results={k: M.per_video(v) for k, v in out.items()})


# ================================================================= GraSP
def grasp(args, model):
    d = json.loads((args.grasp / args.annotations).read_text())
    ann = {x['image_id']: x for x in d['annotations']}
    by_video = defaultdict(list)
    for im in d['images']:
        by_video[im['video_name']].append(im)
    tasks = {'phase': ('phases_categories', 'phases'), 'step': ('steps_categories', 'steps')}
    cats = {t: sorted(d[c], key=lambda x: x['id']) for t, (c, _) in tasks.items()}
    prompts = {t: dict(description=[c['description'] for c in cs], names=[c['name'].replace('_', ' ') for c in cs]) for t, cs in cats.items()}
    queries = {t: {k: model.embed_text(ps, GRASP_INSTRUCTION[t]) for k, ps in v.items()} for t, v in prompts.items()}
    sims, labels = {}, {}
    for v in sorted(by_video)[:args.limit_videos or None]:
        ims = sorted(by_video[v], key=lambda i: i['frame_num'])[:args.limit_frames or None]
        emb = model.embed_windows(model.image_features([str(args.grasp / 'frames' / i['file_name']) for i in ims]), args.window)
        sims[v] = {t: {k: (emb @ q.T).cpu().numpy() for k, q in qs.items()} for t, qs in queries.items()}
        labels[v] = {t: np.array([ann[i['id']][key] for i in ims]) for t, (_, key) in tasks.items()}
        log(video=v, frames=len(ims))
    results = {}
    for t in tasks:
        for k in prompts[t]:
            for mode in ('all', 'no_idle'):  # no_idle: Idle (id 0) frames removed and the Idle query dropped
                per = []
                for v in sims:
                    s, y = sims[v][t][k], labels[v][t][:len(sims[v][t][k])]
                    if mode == 'no_idle':
                        keep = y != 0
                        s, y = s[keep][:, 1:], y[keep] - 1
                    if len(y):
                        per.append(dict(video=v, labels=y, pred=s.argmax(1)))
                results[f'{t}/{k}/{mode}'] = M.per_video(per)
    return dict(window_s=args.window or model.cfg['window_s'], prompts=prompts, results=results)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='task', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--model', type=Path, required=True, help='released model folder')
    common.add_argument('--out', type=Path, required=True)
    common.add_argument('--text-path', help='override the text backbone location (e.g. a local Qwen3-Embedding-8B snapshot)')
    common.add_argument('--limit-videos', type=int, help='smoke test: only the first N videos')
    common.add_argument('--limit-frames', type=int, help='smoke test (cholec80 / grasp): only the first N frames per video')
    p = sub.add_parser('lung', parents=[common])
    p.add_argument('--data', type=Path, default=Path('data/lung'))
    p.add_argument('--workers', type=int, default=8, help='threads decoding clips ahead of the GPU')
    p = sub.add_parser('fullvideo', parents=[common])
    p.add_argument('--data', type=Path, default=Path('data/lung'))
    p.add_argument('--window', type=int, help='seconds per centred window (default: the model config, 64)')
    p.add_argument('--query', default='prototype', choices=('prototype', 'definition'))
    p.add_argument('--smooth', action='store_true', help='also decode the val videos and choose a smoothing on them')
    p = sub.add_parser('cholec80', parents=[common])
    p.add_argument('--frames', type=Path, required=True)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--videos', default='49-80')
    p.add_argument('--official-prompts', help='SurgVLP tests/class_prompt.txt (7 lines)')
    p.add_argument('--window', type=int)
    p = sub.add_parser('grasp', parents=[common])
    p.add_argument('--grasp', type=Path, required=True, help='GraSP_1fps folder (frames/ and annotations/)')
    p.add_argument('--annotations', default='annotations/grasp_long-term_test.json')
    p.add_argument('--window', type=int)
    args = ap.parse_args()
    model = Pipeline.from_pretrained(args.model, text_path=args.text_path)
    result = dict(task=args.task, model=str(args.model), created_at=time.strftime('%Y-%m-%dT%H:%M:%S'),
                  **globals()[args.task](args, model))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str) + '\n')
    if 'summary' in result:
        log(**result['summary'])
    elif 'results' in result:
        log(**{k: dict(acc=round(r['acc_mean'], 4), f1=round(r['f1_mean'], 4)) for k, r in result['results'].items()})
    else:
        log(**{k: {m: v for m, v in result[k].items() if m != 'pooled_bootstrap'} for k in ('unsmoothed', 'smoothed') if k in result})


if __name__ == '__main__':
    main()
