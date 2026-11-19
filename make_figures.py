"""Example segmentations for the report: test images with the filaments of a submission file (no GPU needed).

    python make_figures.py --test-dir path/to/test/test_images

Three full-disk images (with a median, a high and a very high number of filaments) and a zoom on the largest
filament of the last one. Each filament has its own colour and a white outline.
"""
import argparse
import csv
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from pycocotools import mask as mask_utils  # noqa: E402


def read_submission(path):
    # image name -> RLE strings of its filaments
    per_image = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            per_image.setdefault(row['filament_id'].rsplit('_', 1)[0], []).append(row['segmentation_rle'])
    return per_image


def decode(rle, size):
    return mask_utils.decode({'size': list(size), 'counts': rle.encode('utf-8')}).astype(bool)


IMAGE_EXTENSIONS = ('.jpeg', '.jpg', '.png', '.tif', '.tiff', '.bmp')


def index_images(folder):
    # image name (without extension) -> path; any of the usual extensions, sub-folders included
    found = {}
    for root, _, names in os.walk(folder):
        for n in sorted(names):
            stem, ext = os.path.splitext(n)
            if ext.lower() in IMAGE_EXTENSIONS:
                found.setdefault(stem, os.path.join(root, n))
    return found


def overlay(gray, masks, seed=0):
    rgb = np.dstack([gray] * 3).astype(np.float32) / 255.0
    colors = plt.get_cmap('tab20')(np.random.default_rng(seed).permutation(20))[:, :3]
    for i, m in enumerate(masks):
        rgb[m] = 0.35 * rgb[m] + 0.65 * colors[i % 20]
    out = (rgb * 255).astype(np.uint8)
    for m in masks:
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-dir', required=True, help='folder with the test .jpeg images')
    ap.add_argument('--submission', default='submission/submission_scale_stable.csv')
    ap.add_argument('--images', nargs='*', help='image names without extension (default: chosen automatically)')
    ap.add_argument('--out', default='figures/examples.png')
    args = ap.parse_args()

    sub = read_submission(args.submission)
    images = index_images(args.test_dir)
    available = [st for st in sub if st in images]
    if not available:
        raise SystemExit(f'None of the {len(sub)} images of {args.submission} is in {args.test_dir} '
                         f'({len(images)} image files there, e.g. {sorted(images)[:3]}). Check --test-dir.')
    print(f'{len(available)} of the {len(sub)} images of the submission found in {args.test_dir}')
    if args.images:
        stems = [st for st in args.images if st in images]
    else:
        ranked = sorted(available, key=lambda s: (len(sub[s]), s))
        stems = []
        for q in (0.5, 0.75, 0.95):                   # three different images, median to very busy
            i = int(q * (len(ranked) - 1))
            while ranked[i] in stems and i > 0:
                i -= 1
            if ranked[i] not in stems:
                stems.append(ranked[i])

    panels = []
    for st in stems:
        gray = cv2.imread(images[st], cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise SystemExit(f'cannot read {images[st]}')
        masks = [decode(r, gray.shape) for r in sub[st]]
        panels.append((st, gray, masks))

    st, gray, masks = panels[-1]                       # zoom on the largest filament of the last image
    big = max(masks, key=lambda m: m.sum())
    ys, xs = np.nonzero(big)
    cy, cx, half = int(ys.mean()), int(xs.mean()), 300
    y0, x0 = max(0, cy - half), max(0, cx - half)
    y1, x1 = min(gray.shape[0], y0 + 2 * half), min(gray.shape[1], x0 + 2 * half)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, (st_i, g, ms) in zip(axes[:3], panels):
        ax.imshow(overlay(g, ms))
        ax.set_title(f'{st_i}: {len(ms)} filaments', fontsize=10)
        ax.axis('off')
    zoom = overlay(gray[y0:y1, x0:x1], [m[y0:y1, x0:x1] for m in masks])
    axes[3].imshow(zoom)
    axes[3].set_title(f'zoom on {st} ({x1 - x0}x{y1 - y0} px)', fontsize=10)
    axes[3].axis('off')
    axes[2].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='yellow', linewidth=1.5))
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    print('saved', args.out, '| images:', ', '.join(stems))


if __name__ == '__main__':
    main()