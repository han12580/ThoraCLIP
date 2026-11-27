"""Metrics exactly as computed for the paper (copied from the evaluation code)."""
import unicodedata
import warnings

import numpy as np
import torch

PHASE_CLASSES = list(range(8))
MIN_GOLD_POSITIVES = 5


# ---------------------------------------------------------------- clip-level phase
def macro_scores(y, p, classes=PHASE_CLASSES):
    """Macro over classes present in y only (val has no B008, which must not count as F1 = 0)."""
    from sklearn.metrics import balanced_accuracy_score, f1_score
    classes = [c for c in classes if (y == c).any()]
    return dict(classes_evaluated=classes,
                macro_f1=float(f1_score(y, p, labels=classes, average='macro', zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y, p)),
                per_class_f1=[float(v) for v in f1_score(y, p, labels=classes, average=None, zero_division=0)])


def video_bootstrap(y, p, videos, classes=PHASE_CLASSES, rounds=2000, seed=0):
    """95% CI by resampling videos (segments of one video are not independent)."""
    rng = np.random.default_rng(seed)
    unique = np.unique(videos)
    index = {v: np.flatnonzero(videos == v) for v in unique}
    f1, bacc = [], []
    warnings.filterwarnings('ignore', message='y_pred contains classes not in y_true')
    for _ in range(rounds):
        take = np.concatenate([index[v] for v in rng.choice(unique, len(unique))])
        s = macro_scores(y[take], p[take], classes)
        f1.append(s['macro_f1'])
        bacc.append(s['balanced_accuracy'])
    return dict(rounds=rounds, unit='video', macro_f1_95ci=np.percentile(f1, [2.5, 97.5]).tolist(),
                balanced_accuracy_95ci=np.percentile(bacc, [2.5, 97.5]).tolist())


# ---------------------------------------------------------------- steps
def mean_ap(y, scores, classes, legal=None):
    """legal (rows x steps, optional): routing. A blocked row is never retrieved for that step: AP is computed over the
    routed rows and scaled by routed positives / all positives, i.e. blocked positives count in the recall denominator
    but are never found (same as the F1 / Top-1 view). Without legal, plain AP over all rows."""
    from sklearn.metrics import average_precision_score
    present = [c for c in classes if y[:, c].sum() > 0]
    if not present:
        return float('nan'), present

    def ap(c):
        if legal is None:
            return average_precision_score(y[:, c], scores[:, c])
        rows = legal[:, c]
        found = y[rows, c].sum()
        return 0. if found == 0 else average_precision_score(y[rows, c], scores[rows, c]) * found / y[:, c].sum()
    return float(np.mean([ap(c) for c in present])), present


def map_bootstrap(y, scores, videos, classes, rounds=2000, seed=0, legal=None):
    rng = np.random.default_rng(seed)
    unique = np.unique(videos)
    index = {v: np.flatnonzero(videos == v) for v in unique}
    values = []
    for _ in range(rounds):
        take = np.concatenate([index[v] for v in rng.choice(unique, len(unique))])
        values.append(mean_ap(y[take], scores[take], classes, None if legal is None else legal[take])[0])
    return np.nanpercentile(values, [2.5, 97.5]).tolist()


def step_metrics(y, scores, videos, trained, step_ids, legal=None):
    """mAP over trainable steps present in y; 'frequent' = steps with >= 5 positives (CI only for those).
    legal: routing mask (rows x steps); blocked positives are never retrieved (see mean_ap)."""
    from sklearn.metrics import f1_score
    gold_classes = [c for c in trained if y[:, c].sum() > 0]
    ap, present = mean_ap(y, scores, gold_classes, legal)
    frequent = [c for c in gold_classes if y[:, c].sum() >= MIN_GOLD_POSITIVES]
    predicted = (scores >= .5).astype(int)
    return dict(mAP=ap, steps_in_mAP=[step_ids[c] for c in present],
                mAP_frequent=mean_ap(y, scores, frequent, legal)[0],
                mAP_frequent_95ci=map_bootstrap(y, scores, videos, frequent, legal=legal),
                macro_f1_at_0_5_frequent=float(f1_score(y[:, frequent], predicted[:, frequent], average='macro', zero_division=0))
                if frequent else float('nan'),
                frequent_steps=[step_ids[c] for c in frequent],
                gold_positives_not_trainable=int(sum(y[:, c].sum() for c in range(y.shape[1]) if c not in trained)))


def routed_step_metrics(y, step_sims, phase_pred, true_phase, step_parent, videos, trained, step_ids):
    """1+N readout: the routing phase's children keep their similarity; every other step is placed below all legal scores
    (similarities can be negative, so a score of 0 would let an excluded step win Top-1); in a step's AP a blocked clip is
    never retrieved: blocked positives stay in the recall denominator (the same view as Top-1 / F1).
    Routed by the predicted phase (reported) and by the true phase (upper bound)."""
    out = {}
    for name, phases in (('predicted_phase', np.asarray(phase_pred)), ('gold_phase', np.asarray(true_phase))):
        legal = step_parent[None] == phases[:, None]
        floor = (step_sims[legal].min() if legal.any() else 0.) - 1.
        scores = np.where(legal, step_sims, floor)
        m = step_metrics(y, scores, videos, trained, step_ids, legal=legal)
        top = scores.argmax(1)
        hit = (y[np.arange(len(top)), top] == 1) & legal.any(1)  # a clip whose phase has no steps has no Top-1
        m['top1_step_in_gold_steps'] = float(np.mean(hit))
        out[name] = m
    return out


# ---------------------------------------------------------------- retrieval
def caption_key(caption):
    return ' '.join(unicodedata.normalize('NFKC', caption).casefold().split())


def alignment_masks(videos, texts):
    """Rows carry source_id, clip_uid, decode_start_s, decode_end_s, alignment_text. Positive = same clip or identical caption;
    a pair from the same source whose spans overlap by > 50% of the shorter one is not used as a negative."""
    keys = {}
    for row in texts:
        keys.setdefault(row['clip_uid'], set()).add(caption_key(row['alignment_text']))
    text_keys = [caption_key(row['alignment_text']) for row in texts]
    positives, allowances = [], []
    for video in videos:
        positive, allowed = [], []
        for j, text in enumerate(texts):
            pos = video['clip_uid'] == text['clip_uid'] or text_keys[j] in keys.get(video['clip_uid'], set())
            allow = True
            if video['source_id'] == text['source_id']:
                overlap = max(0, min(video['decode_end_s'], text['decode_end_s']) - max(video['decode_start_s'], text['decode_start_s']))
                shorter = min(video['decode_end_s'] - video['decode_start_s'], text['decode_end_s'] - text['decode_start_s'])
                if overlap / shorter > .5:
                    allow = False
            positive.append(pos)
            allowed.append(allow or pos)
        positives.append(positive)
        allowances.append(allowed)
    return torch.tensor(positives, dtype=torch.bool), torch.tensor(allowances, dtype=torch.bool)


def hit_probability(total, positives, draws):
    draws = min(draws, total)
    if draws <= 0:
        return 0.0
    if draws > total - positives:
        return 1.0
    miss = 1.0
    for j in range(draws):
        miss *= (total - positives - j) / (total - j)
    return 1.0 - miss


def retrieval_direction(scores, positive, allowed):
    """Recall@1/5/10 with several positives per query; ties are resolved by their expected hit probability."""
    sums = {k: 0. for k in (1, 5, 10)}
    valid = 0
    for s, p, a in zip(scores, positive, allowed):
        if not (a & ~p).any():
            continue
        best = float(s[p].max())
        above = int(((s > best) & a).sum())
        tied = (s == best) & a
        size, matches = int(tied.sum()), int((tied & p).sum())
        for k in sums:
            sums[k] += hit_probability(size, matches, k - above)
        valid += 1
    if not valid:  # no query has a true negative: report it instead of dividing by zero
        return dict(queries=0, excluded=len(scores), R1=None, R5=None, R10=None)
    return dict(queries=valid, excluded=len(scores) - valid, **{f'R{k}': v / valid for k, v in sums.items()})


# ---------------------------------------------------------------- per-video frame metrics (Cholec80, GraSP, full videos)
def per_video(videos):
    """videos: dicts with labels, pred -> mean / std over videos of frame accuracy and sklearn macro-F1 (classes present in
    that video's labels or predictions), as in the SurgVLP zero-shot protocol."""
    from sklearn.metrics import f1_score
    accs = [float((v['labels'] == v['pred']).mean()) for v in videos]
    f1s = [float(f1_score(v['labels'], v['pred'], average='macro')) for v in videos]
    return dict(videos=len(videos), acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)), f1_mean=float(np.mean(f1s)),
                f1_std=float(np.std(f1s)), per_video=[dict(video=v['video'], acc=a, f1=f) for v, a, f in zip(videos, accs, f1s)])


# ---------------------------------------------------------------- temporal smoothing (boundary-free evaluation)
SMOOTHING_GRID = [('none', 0)] + [('mean', k) for k in (5, 9, 17, 33, 65, 129)] + [('viterbi', r) for r in (0.5, 1, 2, 4, 8, 16, 32, 64)]


def moving(s, k):
    pad = k // 2
    c = np.cumsum(np.pad(s, ((pad + 1, pad), (0, 0)), mode='edge'), 0)
    return (c[k:] - c[:-k]) / k


def viterbi(s, lam):
    n, c = s.shape
    score, back = s[0].copy(), np.zeros((n, c), int)
    for t in range(1, n):
        stay, best = score, score.max()
        arg = int(score.argmax())
        switch = best - lam
        back[t] = np.where(stay >= switch, np.arange(c), arg)
        score = np.maximum(stay, switch) + s[t]
    y = np.zeros(n, int)
    y[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        y[t - 1] = back[t, y[t]]
    return y


def smooth_predict(s, method, value, sigma):
    """none = per-second argmax; mean_k = centred moving average; viterbi_r = switch penalty r x sigma (sigma from val scores)."""
    if method == 'none':
        return s.argmax(1)
    if method == 'mean':
        return moving(s, value).argmax(1)
    return viterbi(s, value * sigma)
