"""Frame decoding used by the paper: 1 fps, short side 512 (aspect kept, even sizes), RGB uint8.

Video files are decoded with ffmpeg (fps filter, frames at k / fps seconds from the start of the decoded range); frame folders
(Cholec80, GraSP) are read with PIL and resized the same way.
"""
import shutil
import subprocess

import numpy as np

FFMPEG = shutil.which('ffmpeg') or 'ffmpeg'
FFPROBE = shutil.which('ffprobe') or 'ffprobe'


def probe(path, ffprobe=FFPROBE):
    """(width, height, duration_s) of the first video stream."""
    out = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height:format=duration',
                          '-of', 'default=noprint_wrappers=1', str(path)], capture_output=True, text=True, check=True).stdout
    kv = dict(line.split('=', 1) for line in out.splitlines() if '=' in line)
    return int(kv['width']), int(kv['height']), float(kv.get('duration') or 'nan')


def decode(path, start=0., duration=None, fps=1., short_side=512, ffmpeg=FFMPEG, ffprobe=FFPROBE):
    """Frames [T, H, W, 3] uint8 and their times [T] in seconds from `start`.

    duration=None decodes to the end of the file. Same ffmpeg command as the paper's decoder."""
    width, height, total = probe(path, ffprobe)
    if duration is None:
        duration = total - start + 1.
    scale = short_side / min(width, height)
    w, h = round(width * scale / 2) * 2, round(height * scale / 2) * 2
    seek = ['-ss', f'{start:.3f}'] if start else []
    command = [ffmpeg, '-nostdin', '-v', 'error', '-threads', '2', *seek, '-t', f'{duration:.3f}', '-i', str(path), '-an',
               '-vf', f'fps={fps}:start_time=0:round=near,scale={w}:{h}:flags=bilinear',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    result = subprocess.run(command, capture_output=True, timeout=7200)
    count = len(result.stdout) // (w * h * 3)
    if result.returncode or count == 0:
        raise RuntimeError(f'Decode failed for {path}: {result.stderr.decode(errors="replace")[-500:]}')
    frames = np.frombuffer(result.stdout[:count * w * h * 3], np.uint8).reshape(count, h, w, 3)
    return frames, (np.arange(count) / fps).astype(np.float32)


def read_images(paths, short_side=512):
    """Frame files -> list of uint8 arrays with the short side resized to `short_side` (PIL bilinear, even sizes)."""
    from PIL import Image
    frames = []
    for p in paths:
        im = Image.open(p).convert('RGB')
        if short_side:
            w, h = im.size
            s = short_side / min(w, h)
            im = im.resize((round(w * s / 2) * 2, round(h * s / 2) * 2), Image.BILINEAR)
        frames.append(np.asarray(im))
    return frames
