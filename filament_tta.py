"""Test-time augmentation and mask voting for solar filament instance segmentation.

Two YOLO segmentation models trained by hdjojo (Kaggle Models, Apache 2.0) each predict 8 views of every H-alpha
image: the 4 mirror images (original, left-right, up-down, both) at 2048 px and the same 4 enlarged to 2560 px.
Masks of the same filament from different views are grouped, fused by confidence-weighted pixel voting, ranked by
consensus x shape stability, and made disjoint. The best TARGET filaments over the whole test set are written in the
competition format (COCO RLE). This is the code that produced the final submission (public PQ 0.44, private 0.42).
"""
import csv
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils

VIEWS = [(0, False), (0, True), (2, True), (2, False)]  # (quarter turns, mirror): original, left-right, up-down, both
BIG = 2560          # second image size: the same 4 views enlarged 1.25x
FLOOR = 0.15        # predictions below this confidence are not collected
CLUSTER_IOU = 0.4   # masks of different views overlapping more than this are the same filament
MIN_AREA = 30       # pixels; smaller masks are ignored
KEEP_FRAC = 0.5     # a filament that loses more than half of its area to better-ranked ones is dropped
MIN_SCORE = 0.03    # fused filaments below this consensus score are never used
TARGET = 1317       # filaments kept over the test set: the number the two models give at confidence 0.5
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def mask_to_rle(binary_mask):
    """COCO run-length encoding (column-major), as expected by the competition."""
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    return rle['counts'].decode('utf-8')


def counts(X, chunk=4):
    """Pixels of each mask in X, in small chunks (torch.sum on booleans makes an 8-bytes-per-pixel copy)."""
    if len(X) == 0:
        return torch.zeros(0, dtype=torch.long, device=X.device)
    return torch.cat([torch.count_nonzero(X[i:i + chunk].flatten(1), dim=1) for i in range(0, len(X), chunk)])


def predict_masks(model, cfg, img, size, k, lr):
    """Masks (bool, aligned with the original image) and confidences of one model on one view, or None."""
    im = np.rot90(img, k) if k else img
    im = im[:, ::-1] if lr else im
    r = model.predict(np.ascontiguousarray(im), imgsz=size, conf=FLOOR, iou=cfg.IOU, max_det=cfg.MAX_DET,
                      verbose=False)[0]
    if r.masks is None or len(r.boxes) == 0:
        return None
    m = r.masks.data
    if tuple(m.shape[-2:]) != im.shape[:2]:          # enlarged views: back to the image size
        m = torch.cat([F.interpolate(m[i:i + 8, None].float(), size=im.shape[:2], mode='bilinear',
                                     align_corners=False)[:, 0] > 0.5 for i in range(0, len(m), 8)])
    else:
        m = m > 0.5
    m = m.to(DEV)
    c = r.boxes.conf.float().to(DEV)
    big = counts(m) >= MIN_AREA
    if not bool(big.any()):
        return None
    m, c = m[big], c[big]
    if lr:
        m = m.flip(-1)
    if k:
        m = torch.rot90(m, -k, dims=(-2, -1))
    return m.contiguous(), c


def pair_iou(M):
    """Exact IoU of every pair of masks whose bounding boxes overlap (0 for the others)."""
    n, h, w = M.shape
    ys, xs = M.any(2), M.any(1)
    y0, y1 = ys.float().argmax(1), h - ys.flip(1).float().argmax(1)
    x0, x1 = xs.float().argmax(1), w - xs.flip(1).float().argmax(1)
    area = counts(M).float()
    ov = (y0[:, None] < y1[None]) & (y0[None] < y1[:, None]) & (x0[:, None] < x1[None]) & (x0[None] < x1[:, None])
    iou = torch.zeros((n, n), device=M.device)
    b = list(zip(y0.tolist(), y1.tolist(), x0.tolist(), x1.tolist()))
    for i in range(n):
        js = torch.nonzero(ov[i]).flatten()
        js = js[js > i]
        a0, a1, c0, c1 = b[i]
        for q in range(0, len(js), 16):
            jq = js[q:q + 16]
            inter = counts(M[jq, a0:a1, c0:c1] & M[i, a0:a1, c0:c1]).float()
            v = inter / (area[i] + area[jq] - inter).clamp(min=1)
            iou[i, jq] = v
            iou[jq, i] = v
    return iou.cpu().numpy()


def clusters_of(preds):
    """Group the masks of all views -> [(score, stability, fused mask)].

    score = sum of the members' confidences / number of views (a filament found by every view with confidence c
    scores c); stability = confidence-weighted mean IoU between the members and the fused mask.
    """
    n_src = len(preds)
    items = [(k, p) for k, p in enumerate(preds) if p is not None]
    if not items:
        return []
    M = torch.cat([p[0] for _, p in items])
    C = torch.cat([p[1] for _, p in items])
    S = np.concatenate([np.full(len(p[1]), k) for k, p in items])
    iou = pair_iou(M)
    conf = C.cpu().numpy()
    used, groups = np.zeros(len(conf), bool), []
    for i in np.argsort(-conf, kind='stable'):        # greedy: the most confident free mask starts a group
        if used[i]:
            continue
        used[i], members = True, [i]
        for s in sorted(set(S.tolist()) - {S[i]}):     # at most one mask per view
            cand = np.flatnonzero(~used & (S == s) & (iou[i] > CLUSTER_IOU))
            if len(cand):
                j = cand[np.argmax(iou[i, cand])]
                used[j] = True
                members.append(j)
        groups.append((float(conf[members].sum()) / n_src, members))
    groups.sort(key=lambda g: -g[0])
    out = []
    for score, members in groups:
        if score < MIN_SCORE:
            break
        idx = torch.as_tensor(members, device=DEV)
        wt = C[idx]
        f = M[idx[0]] if len(members) == 1 else (M[idx].float() * wt[:, None, None]).sum(0) >= 0.5 * wt.sum()
        area_f = int(torch.count_nonzero(f))
        if area_f < MIN_AREA:
            continue
        inter = counts(M[idx] & f).float()
        iou_m = inter / (counts(M[idx]).float() + area_f - inter).clamp(min=1)
        out.append((score, float((iou_m * wt).sum() / wt.sum()), f))
    return out


def paint(cl, stable, h, w):
    """Disjoint filaments in ranking order -> [(ranking score, rle)]; stable=True ranks by score x stability."""
    key = (lambda c: c[0] * c[1]) if stable else (lambda c: c[0])
    order = sorted(range(len(cl)), key=lambda i: -key(cl[i]))
    taken = torch.zeros((h, w), dtype=torch.bool, device=DEV)
    kept = []
    for i in order:
        f = cl[i][2]
        a0 = int(torch.count_nonzero(f))
        m = f & ~taken
        a = int(torch.count_nonzero(m))
        if a < MIN_AREA or a < KEEP_FRAC * a0:
            continue
        taken |= m
        kept.append((key(cl[i]), mask_to_rle(m.to(torch.uint8).cpu().numpy())))
    return kept


def predict_image(models, img):
    """All filaments of one image (BGR array) -> [(ranking score, rle)], best first.

    models: [(ultralytics YOLO model, config with IMG_SIZE, IOU, MAX_DET), ...]
    """
    h, w = img.shape[:2]
    pa = [predict_masks(model, cfg, img, cfg.IMG_SIZE, k, lr) for model, cfg in models for k, lr in VIEWS]
    try:
        pb = [predict_masks(model, cfg, img, BIG, k, lr) for model, cfg in models for k, lr in VIEWS]
        cl = clusters_of(pa + pb)
    except Exception as e:      # e.g. out of GPU memory at 2560 px: this image keeps the 2048 px views only
        print('enlarged views failed:', repr(e)[:150])
        if DEV == 'cuda':
            torch.cuda.empty_cache()
        cl = clusters_of(pa)
    return paint(cl, True, h, w)


def select_top(results, count=TARGET):
    """The `count` best filaments over all images -> (rows [(filament_id, rle)], score threshold)."""
    scores = np.sort(np.array([s for v in results.values() for s, _ in v]))[::-1]
    tau = float(scores[min(count, len(scores)) - 1]) if len(scores) and count > 0 else float('inf')
    rows = [(f'{st}_{i}', rle) for st in sorted(results)
            for i, (s, rle) in enumerate([c for c in results[st] if c[0] >= tau], 1)]
    return rows, tau


def write_submission(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['filament_id', 'segmentation_rle'])
        wr.writerows(rows)


def read_submission(path):
    """image stem -> list of RLE strings"""
    per_image = defaultdict(list)
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            per_image[row['filament_id'].rsplit('_', 1)[0]].append(row['segmentation_rle'])
    return per_image


def decode(rle, size=(2048, 2048)):
    return mask_utils.decode({'size': list(size), 'counts': rle.encode('utf-8')}).astype(bool)


def compare_submissions(path_a, path_b, size=(2048, 2048)):
    """How close two submission files are: identical masks, and for the others the best IoU with a mask of b."""
    a, b = read_submission(path_a), read_submission(path_b)
    n_a, n_b = sum(map(len, a.values())), sum(map(len, b.values()))
    same, ious = 0, []
    for st in sorted(set(a) | set(b)):
        ra, rb = list(a.get(st, [])), list(b.get(st, []))
        rest_a = []
        for r in ra:
            if r in rb:
                rb.remove(r)
                same += 1
            else:
                rest_a.append(r)
        mb = [decode(r, size) for r in rb]
        for r in rest_a:
            m = decode(r, size)
            ious.append(max([(m & x).sum() / max(1, (m | x).sum()) for x in mb] or [0.0]))
    return dict(filaments_a=n_a, filaments_b=n_b, identical=same, different=len(ious),
                mean_best_iou_of_different=float(np.mean(ious)) if ious else None)
