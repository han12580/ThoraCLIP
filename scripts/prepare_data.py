#!/usr/bin/env python3
"""Turn the public annotation release + your own copies of the source videos into the layout the training and
evaluation scripts read.

Inputs
  --release  the released dataset folder (clips/{train,val,test}.jsonl, videos.jsonl, taxonomy.json, gold/)
  --videos   a folder with one file per source video, named <video_id>.<ext> (mp4/mkv/webm/mov); obtain the videos
             from the source pages listed in videos.jsonl under each site's terms of use. Videos are not redistributed.
Outputs (under --out, default data/lung)
  in_scope/{train,val,test}.jsonl   clip rows with phase_target / step_targets and media/clips/<clip_id>.mp4
  in_scope/catalog.json             class list (id, level, name_en, name_zh, definitions)
  catalog.json                      doctor-facing catalog (phases / steps with name_zh, definition_zh, parent)
  videos.jsonl                      per video: media/videos/<video_id>.<ext> link and labelled intervals
  gold/gold_{segments,tasks}.jsonl  copies of the doctor gold benchmark (segments: --gold-segments; tasks: agreement)
  media/clips/*.mp4                 clips cut with the same ffmpeg settings as the paper (libx264 crf 18, aac)

Missing videos are reported and their clips left out; the summary prints how many clips are usable.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
EXTS = ('.mp4', '.mkv', '.webm', '.mov', '.m4v')


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def find_video(folder, video_id):
    for ext in EXTS:
        p = folder / f'{video_id}{ext}'
        if p.exists():
            return p
    return None


def cut(ffmpeg, src, start, duration, dst, threads):
    if dst.exists() and dst.stat().st_size > 0:
        return 'exists'
    tmp = dst.with_name(dst.stem + '.partial.mp4')
    subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-y', '-ss', f'{start}', '-i', str(src), '-t', f'{duration}',
                    '-map', '0:v:0', '-map', '0:a:0?', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                    '-threads', str(threads), '-c:a', 'aac', '-movflags', '+faststart', str(tmp)], check=True)
    tmp.replace(dst)
    return 'cut'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--release', type=Path, required=True)
    ap.add_argument('--videos', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'data/lung')
    ap.add_argument('--ffmpeg', default=shutil.which('ffmpeg') or 'ffmpeg')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--threads', type=int, default=4, help='libx264 threads per clip')
    a = ap.parse_args()
    rel, out = a.release, a.out
    tax = json.loads((rel / 'taxonomy.json').read_text(encoding='utf-8'))
    phase_index = {p['id']: i for i, p in enumerate(tax['phases'])}
    step_index = {s['id']: i for i, s in enumerate(tax['steps'])}

    videos = {v['video_id']: v for v in jsonl(rel / 'videos.jsonl')}
    found = {vid: find_video(a.videos, vid) for vid in videos}
    missing = sorted(v for v, p in found.items() if p is None)
    (out / 'media/clips').mkdir(parents=True, exist_ok=True)
    (out / 'media/videos').mkdir(parents=True, exist_ok=True)
    for vid, p in found.items():
        if p is not None:
            link = out / 'media/videos' / f'{vid}{p.suffix}'
            if not link.exists():
                os.symlink(p.resolve(), link)

    rows, jobs = {}, []
    for split in ('train', 'val', 'test'):
        rows[split] = []
        for c in jsonl(rel / f'clips/{split}.jsonl'):
            if found[c['video_id']] is None:
                continue
            targets = [0] * len(step_index)
            for s in c['step_ids']:
                targets[step_index[s]] = 1
            rows[split].append(dict(
                clip_id=c['clip_id'], video_id=c['video_id'], dataset=c['source'], split=split, group_id=c['group_id'],
                video=f"media/clips/{c['clip_id']}.mp4", start_s=c['start_s'], end_s=c['end_s'],
                source_start_s=c['start_s'], source_end_s=c['end_s'], duration_s=c['duration_s'],
                phase_id=c['phase_id'], phase_ids=[c['phase_id']], phase_target=phase_index[c['phase_id']],
                step_ids=c['step_ids'], step_targets=targets, phase_training_eligible=True,
                step_training_eligible=bool(c['step_ids']),
                caption=c['caption'], description=c['description'], paraphrases=c['paraphrases'],
                title=videos[c['video_id']]['title'], source_url=videos[c['video_id']]['source_url'],
                label_status=c['label_status']))
            jobs.append((found[c['video_id']], c['start_s'], c['duration_s'], out / 'media/clips' / f"{c['clip_id']}.mp4"))

    with ThreadPoolExecutor(a.workers) as pool:
        results = list(pool.map(lambda j: cut(a.ffmpeg, *j, a.threads), jobs))

    for split, rs in rows.items():
        write_jsonl(out / f'in_scope/{split}.jsonl', rs)
    if not (out / 'in_scope/media').exists():  # clip paths are relative to in_scope/ (the --dataset folder of the scripts)
        os.symlink('../media', out / 'in_scope/media')
    classes = ([dict(id=p['id'], level='phase', name_en=p['name'], name_zh=p['name_zh'], definition_en=p['definition'],
                     definition_zh=p['definition_zh']) for p in tax['phases']]
               + [dict(id=s['id'], level='step', parent=s['phase_id'], name_en=s['name'], name_zh=s['name_zh'],
                       definition_en=s['definition'], definition_zh=s['definition_zh']) for s in tax['steps']])
    (out / 'in_scope/catalog.json').write_text(json.dumps(dict(version=tax['version'], classes=classes), ensure_ascii=False, indent=1),
                                               encoding='utf-8')
    (out / 'catalog.json').write_text(json.dumps(dict(
        phases=[dict(id=p['id'], name_zh=p['name_zh'], definition_zh=p['definition_zh']) for p in tax['phases']],
        steps=[dict(id=s['id'], name_zh=s['name_zh'], definition_zh=s['definition_zh'], parent=s['phase_id']) for s in tax['steps']]),
        ensure_ascii=False, indent=1), encoding='utf-8')
    by_video = {}
    for split, rs in rows.items():
        for r in rs:
            by_video.setdefault(r['video_id'], []).append(r)
    write_jsonl(out / 'videos.jsonl', [dict(
        video_id=vid, group_id=v['group_id'], split=v['split'], video=f"media/videos/{vid}{found[vid].suffix}", title=v['title'],
        duration_s=v['duration_s'], dense_frame_annotations=False, gaps_are_unlabelled=True,
        intervals=[dict(clip_id=r['clip_id'], start_s=r['start_s'], end_s=r['end_s'], phase_id=r['phase_id'], step_ids=r['step_ids'])
                   for r in sorted(by_video.get(vid, []), key=lambda r: r['start_s'])])
        for vid, v in sorted(videos.items()) if found[vid] is not None])
    (out / 'gold').mkdir(exist_ok=True)
    for name in ('gold_segments.jsonl', 'gold_tasks.jsonl'):
        shutil.copyfile(rel / 'gold' / name, out / 'gold' / name)
    summary = dict(videos_found=len(videos) - len(missing), videos_missing=len(missing),
                   clips={s: len(r) for s, r in rows.items()}, cut=results.count('cut'), reused=results.count('exists'))
    (out / 'prepare_summary.json').write_text(json.dumps(dict(summary, missing_videos=missing), indent=1))
    print(json.dumps(summary, indent=1))
    if missing:
        print(f'{len(missing)} videos missing (see prepare_summary.json); their clips and gold segments are not usable.')


if __name__ == '__main__':
    main()
