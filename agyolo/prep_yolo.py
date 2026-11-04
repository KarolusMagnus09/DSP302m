"""Bước 1: chuyển nhãn Action Genome (ag_index.pkl) sang định dạng YOLO.

  python -m agyolo.prep_yolo --index ag_index.pkl --frames frames --out yolo_data --max_train 60000 --val_n 1500

36 lớp: 0 = person, 1..35 = 35 vật AG (cùng thứ tự agsgg.constants.OBJ_CLASSES).
- Dùng lại ĐÚNG ag_index.pkl (train/dev) nên cùng cách chia video với baseline agsgg; test không đụng tới.
- Frame không có bbox người bị BỎ khỏi tập train YOLO (người có mặt nhưng không được gán nhãn -> sẽ dạy detector
  coi người là nền). Muốn giữ: --keep_noperson.
- Không copy ảnh: tạo thư mục <out>/images trỏ tới thư mục frames (symlink; Windows dùng junction;
  không được thì copy đúng các file cần dùng). YOLO suy ra nhãn từ đường dẫn ".../images/..." -> ".../labels/...".
"""
import argparse, json, os, random, shutil, subprocess

import pickle

from agsgg.constants import OBJECTS
from agsgg.progress import tqdm, write


def link_dir(src, dst):
    """Trả về cách đã dùng: symlink | junction | exists | None (thất bại -> phải copy)."""
    if os.path.lexists(dst):
        return 'exists'
    src = os.path.abspath(src)
    try:
        os.symlink(src, dst, target_is_directory=True)
        return 'symlink'
    except (OSError, NotImplementedError):
        pass
    if os.name == 'nt':
        r = subprocess.run(['cmd', '/c', 'mklink', '/J', dst, src], capture_output=True)
        if r.returncode == 0:
            return 'junction'
    return None


def yolo_lines(f):
    w, h = f['size']
    lines = []

    def add(c, x1, y1, x2, y2):
        x1, x2 = max(0.0, min(float(x1), w)), max(0.0, min(float(x2), w))
        y1, y2 = max(0.0, min(float(y1), h)), max(0.0, min(float(y2), h))
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        if bw > 0.002 and bh > 0.002:
            lines.append(f'{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {bw:.6f} {bh:.6f}')

    if f['has_person']:
        add(0, *f['person'])
    for l, b in zip(f['labels'], f['boxes']):
        add(int(l) + 1, *b)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', default='ag_index.pkl')
    ap.add_argument('--frames', required=True, help='<frames>/<video.mp4>/<000123.png>')
    ap.add_argument('--out', default='yolo_data')
    ap.add_argument('--max_train', type=int, default=0, help='>0: chỉ lấy ngẫu nhiên n frame train (tiết kiệm thời gian)')
    ap.add_argument('--val_n', type=int, default=1500, help='số frame dev dùng để validate YOLO')
    ap.add_argument('--keep_noperson', action='store_true')
    ap.add_argument('--copy', action='store_true', help='copy ảnh thay vì symlink/junction')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    index = pickle.load(open(a.index, 'rb'))
    rng = random.Random(a.seed)

    def collect(split, limit):
        items = [(v, f) for v, frs in index[split].items() for f in frs if f['has_person'] or a.keep_noperson]
        if limit and limit < len(items):
            items = rng.sample(items, limit)
        return sorted(items, key=lambda t: (t[0], t[1]['frame']))

    sets = {'train': collect('train', a.max_train), 'val': collect('dev', a.val_n)}
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    img_root = os.path.join(out, 'images')
    how = None if a.copy else link_dir(a.frames, img_root)
    if how is None:
        write('không tạo được symlink/junction -> copy các frame cần dùng (có thể lâu, tốn dung lượng)')
        how = 'copy'
    write(f'ảnh: {how}')

    for name, items in sets.items():
        paths = []
        for v, f in tqdm(items, desc=f'nhãn {name}', unit='frame'):
            stem = os.path.splitext(f['frame'])[0]
            ldir = os.path.join(out, 'labels', v)
            os.makedirs(ldir, exist_ok=True)
            with open(os.path.join(ldir, stem + '.txt'), 'w') as fh:
                fh.write('\n'.join(yolo_lines(f)))
            ip = os.path.join(img_root, v, f['frame'])
            if how == 'copy':
                os.makedirs(os.path.dirname(ip), exist_ok=True)
                if not os.path.exists(ip):
                    shutil.copy2(os.path.join(a.frames, v, f['frame']), ip)
            paths.append(ip.replace('\\', '/'))
        with open(os.path.join(out, f'{name}.txt'), 'w') as fh:
            fh.write('\n'.join(paths))
        write(f'{name}: {len(items)} frame')

    names = '\n'.join(f'  {i}: {n}' for i, n in enumerate(OBJECTS))
    with open(os.path.join(out, 'data.yaml'), 'w', encoding='utf-8') as fh:
        fh.write(f"path: {out.replace(chr(92), '/')}\ntrain: train.txt\nval: val.txt\nnames:\n{names}\n")
    json.dump(dict(index=a.index, max_train=a.max_train, val_n=a.val_n, link=how, seed=a.seed,
                   n_train=len(sets['train']), n_val=len(sets['val'])), open(os.path.join(out, 'meta.json'), 'w'))
    write(f'xong -> {os.path.join(out, "data.yaml")}')


if __name__ == '__main__':
    main()
