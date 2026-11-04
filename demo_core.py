"""Suy luận video: YOLO (phát hiện người + vật) + STTran (quan hệ người–vật) -> video có box và triplet.

Dùng chung cho app Gradio và cho chạy dòng lệnh (python demo_core.py --video in.mp4 --out out.mp4).
Cần thư mục `agyolo/` và `agsgg/` (chỉ file .py) nằm cạnh file này.
"""
import argparse, collections, os, shutil, subprocess, tempfile, time

import logging

import cv2
import numpy as np
import torch

from agsgg.constants import OBJ_CLASSES, ATTENTION, SPATIAL, CONTACTING, N_SPA, N_CON
from agyolo.extract_feats import roi_feats, union, select_dets
from agyolo.sttran import STTran, build_clip

FONT = cv2.FONT_HERSHEY_SIMPLEX
HIDE = {'not_contacting', 'not_looking_at', 'unsure'}      # quan hệ "phủ định", không đáng hiện


def put(img, text, org, scale, color, th=1):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, th, cv2.LINE_AA)


def col(c):
    return tuple(int(v) for v in np.random.RandomState(c).randint(70, 255, 3))


def to_h264(src, dst):
    """mp4v không phát được trong trình duyệt -> chuyển H.264 nếu có ffmpeg (hệ thống hoặc imageio-ffmpeg)."""
    exe = shutil.which('ffmpeg')
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            exe = None
    if exe:
        r = subprocess.run([exe, '-y', '-loglevel', 'error', '-i', src, '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                            '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', dst], capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(dst):
            os.remove(src)
            return dst
        print('[to_h264] ffmpeg lỗi, giữ bản mp4v (có thể không phát được trong trình duyệt):', r.stderr.strip()[:500])
    else:
        print('[to_h264] không tìm thấy ffmpeg, giữ bản mp4v (có thể không phát được trong trình duyệt)')
    shutil.move(src, dst)
    return dst


def find_ffmpeg():
    """Ưu tiên ffmpeg của imageio-ffmpeg (có libx264), sau đó mới tới ffmpeg hệ thống."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which('ffmpeg')


def probe_video(path):
    """Mô tả ngắn file video đầu ra (dung lượng + dòng 'Video:' của ffmpeg) để chẩn đoán lỗi không phát được."""
    try:
        size = os.path.getsize(path) / 1e6
    except OSError:
        return f'{path}: KHÔNG TỒN TẠI'
    line = '?'
    exe = find_ffmpeg()
    if exe:
        r = subprocess.run([exe, '-hide_banner', '-i', path], capture_output=True, text=True)
        for ln in (r.stderr or '').splitlines():
            if 'Video:' in ln:
                line = ln.strip()
                break
    return f'{path} | {size:.1f} MB | {line}'


class VideoSink:
    """Ghi video H.264 (yuv420p, faststart) bằng cách đẩy frame BGR thẳng vào ffmpeg -> trình duyệt phát được.
    Không có ffmpeg thì ghi mp4v bằng OpenCV (trình duyệt có thể không phát được)."""

    def __init__(self, path, fps, w, h):
        self.path, self.w, self.h = path, w, h
        self.exe = find_ffmpeg()
        self.p = self.cv = None
        if self.exe:
            self.err = tempfile.TemporaryFile()
            cmd = [self.exe, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}',
                   '-r', f'{fps:.4f}', '-i', '-', '-an', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                   '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', path]
            self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self.err)
        else:
            print('[video] không tìm thấy ffmpeg -> ghi mp4v, trình duyệt có thể không phát được')
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            if not self.cv.isOpened():
                raise RuntimeError(f'Không tạo được file video đầu ra: {path}')

    def write(self, frame):
        if frame.shape[0] != self.h or frame.shape[1] != self.w:
            frame = cv2.resize(frame, (self.w, self.h))
        if self.p is not None:
            try:
                self.p.stdin.write(np.ascontiguousarray(frame).tobytes())
            except (BrokenPipeError, OSError):
                self.release()                                  # ffmpeg đã chết -> ném lỗi kèm nội dung stderr
        else:
            self.cv.write(frame)

    def release(self):
        if self.p is not None:
            try:
                self.p.stdin.close()
            except Exception:
                pass
            rc = self.p.wait()
            self.err.seek(0)
            msg = self.err.read().decode('utf-8', 'ignore').strip()
            self.err.close()
            self.p = None
            if rc != 0:
                raise RuntimeError(f'ffmpeg lỗi (mã {rc}): {msg[:600]}')
        elif self.cv is not None:
            self.cv.release()
            self.cv = None


class Demo:
    def __init__(self, yolo_w, stt_w, imgsz=640, device=None):
        from ultralytics import YOLO
        logging.getLogger('ultralytics').setLevel(logging.ERROR)     # bớt cảnh báo lặp mỗi frame
        self.dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.half = self.dev.startswith('cuda') and os.environ.get('HALF', '1') != '0'   # HALF=0 -> chạy fp32
        self.S = imgsz
        self.yolo = YOLO(yolo_w)
        # Ngưỡng conf riêng theo lớp, vd. CLASS_CONF="chair=0.03,table=0.05" (mặc định chỉ hạ ngưỡng của ghế; CLASS_CONF= để tắt)
        name2id = {str(v).lower(): int(k) for k, v in self.yolo.names.items()}
        self.cls_conf = {}
        for it in os.environ.get('CLASS_CONF', 'chair=0.03').split(','):
            if '=' not in it:
                continue
            nm, v = it.split('=', 1)
            nm = nm.strip().lower()
            if nm in name2id:
                self.cls_conf[name2id[nm]] = float(v)
            else:
                print(f'[CLASS_CONF] lớp "{nm}" không có trong mô hình, bỏ qua. Các lớp: {sorted(name2id)}')
        self.store = {}
        self.yolo.model.model[-1].register_forward_pre_hook(lambda m, inp: self.store.__setitem__('f', list(inp[0])))
        ck = torch.load(stt_w, map_location=self.dev, weights_only=False)
        self.model = STTran(**ck['cfg']).to(self.dev)
        self.model.load_state_dict(ck['model'])
        self.model.eval()
        self.Wn, self.D = ck['cfg'].get('window', 2), ck['cfg']['feat_dim']

    def _tick(self):
        if self.dev.startswith('cuda'):
            torch.cuda.synchronize()
        return time.perf_counter()

    @torch.inference_mode()
    def run(self, video, out, conf=0.25, ctx_sec=0.5, rel_thr=0.30, topk=4, max_obj=12, max_sec=0, progress=None):
        S, dev = self.S, self.dev
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise RuntimeError(f'Không mở được video: {video}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not (fps and np.isfinite(fps) and 1 <= fps <= 240):          # video webcam/trình duyệt hay báo fps sai (0, nan, 1000...)
            fps = 30.0
        nt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        n_total = int(nt) if (nt and np.isfinite(nt) and 0 < nt < 1e9) else 0   # không rõ số frame -> bỏ thanh tiến trình
        # Lấy kích thước từ FRAME THẬT, không tin CAP_PROP_*: video quay bằng điện thoại (có cờ xoay) thì
        # OpenCV tự xoay frame nhưng CAP_PROP vẫn báo w,h cũ -> VideoWriter lặng lẽ bỏ hết frame -> video rỗng.
        ok, first = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError('Video không đọc được frame nào (file hỏng hoặc codec không được OpenCV hỗ trợ).')
        h, w = first.shape[:2]
        pending = first          # KHÔNG tua về đầu (video thiếu metadata/thời lượng thì seek hỏng): dùng lại frame đã đọc
        step = max(1, round(ctx_sec * fps))
        writer = VideoSink(out, fps, w, h)
        r = S / max(h, w)
        nw, nh = max(1, round(w * r)), max(1, round(h * r))
        l, t = (S - nw) // 2, (S - nh) // 2
        off = np.array([l, t, l, t], np.float32)
        nz = np.array([w, h, w, h], np.float32)
        hist = collections.deque(maxlen=max(step * (self.Wn - 1), 1))
        fs = max(0.45, w / 1280 * 0.6)
        th = max(1, round(fs * 2))
        lh = int(34 * fs)
        i, t_yolo, t_rel, n_bad = 0, 0.0, 0.0, 0
        t_all = time.perf_counter()
        seen = collections.Counter()
        segs, obj_cnt = {}, collections.Counter()       # gom triplet liên tiếp thành đoạn thời gian để đưa cho LLM
        gap = max(1, round(0.5 * fps))                 # cách nhau <= 0.5 s thì coi là cùng một đoạn
        while True:
            if pending is not None:
                ok, frame, pending = True, pending, None
            else:
                ok, frame = cap.read()
            if not ok or (max_sec and i >= max_sec * fps):
                break
            t0 = self._tick()
            canvas = np.full((S, S, 3), 114, np.uint8)
            canvas[t:t + nh, l:l + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            x = torch.from_numpy(np.ascontiguousarray(canvas[:, :, ::-1])).permute(2, 0, 1)[None].to(dev).float() / 255
            res = self.yolo.predict(x, imgsz=S, conf=min([conf] + list(self.cls_conf.values())), iou=0.6, max_det=100,
                                    half=self.half, device=dev, verbose=False)
            bd = res[0].boxes.data.cpu()
            if len(bd):
                thr = torch.full((len(bd),), float(conf))
                cid = bd[:, 5].long()
                for c_, t_ in self.cls_conf.items():
                    thr[cid == c_] = min(float(conf), t_)             # ngưỡng riêng của lớp (chỉ để HẠ, không nâng)
                bd = bd[bd[:, 4] >= thr]
            person, ob = select_dets(bd, max_obj)
            if ob is not None and len(ob):
                good = torch.isfinite(ob[:, :5]).all(1)           # YOLO fp16 đôi khi cho box NaN/inf
                if not good.all():
                    n_bad += int((~good).sum())
                    ob = ob[good]
            if person is not None and not torch.isfinite(person[:5]).all():
                n_bad += 1
                person = None
            t1 = self._tick()
            t_yolo += t1 - t0

            rec, rels, rels_all = None, [], []
            if person is not None and len(ob):
                n = len(ob)
                pb = person[:4][None]
                obx = ob[:, :4]
                boxes = torch.cat([pb, obx, union(pb.expand(n, 4), obx)], 0)
                F_ = roi_feats(self.store['f'], S, torch.zeros(2 * n + 1, dtype=torch.long, device=dev), boxes.to(dev)).half().cpu()
                pbox_img = (pb[0].numpy() - off) / r
                obox_img = (obx.numpy() - off) / r
                rec = dict(n=n, per=F_[0].clone(), obj=F_[1:1 + n].clone(), uni=F_[1 + n:].clone(),
                           boxes=np.clip(obox_img / nz, 0, 1).astype(np.float32),
                           labels=(ob[:, 5].long() - 1).numpy().astype(np.int16),
                           scores=ob[:, 4].numpy().astype(np.float32),
                           pbox=np.clip(pbox_img / nz, 0, 1).astype(np.float32), pscore=float(person[4]))
                ctx = []
                for k in range(self.Wn - 1, 0, -1):
                    idx = len(hist) - step * k
                    if idx >= 0 and hist[idx] is not None:
                        ctx.append(hist[idx])
                frames = [dict(rec=c, target=None, att=np.full(c['n'], -1, np.int64),
                               spa=np.zeros((c['n'], N_SPA), np.float32), con=np.zeros((c['n'], N_CON), np.float32),
                               matched=np.zeros(c['n'], bool)) for c in ctx + [rec]]
                clip, _ = build_clip([dict(vid='demo', frames=frames, D=self.D)])
                o = self.model(clip.to(dev))
                sel = clip.fidx == len(frames) - 1
                att = o['att'][sel].softmax(-1).cpu().numpy()
                spa = o['spa'][sel].sigmoid().cpu().numpy()
                con = o['con'][sel].sigmoid().cpu().numpy()
                for j in range(n):
                    for names, pr in ((ATTENTION, att[j]), (SPATIAL, spa[j]), (CONTACTING, con[j])):
                        kk = int(pr.argmax())
                        if names[kk] not in HIDE and pr[kk] >= rel_thr:
                            rels.append((float(pr[kk]) * float(rec['scores'][j]), j, names[kk]))
                rels_all = sorted(rels, reverse=True)
                rels = rels_all[:topk]
            hist.append(rec)
            t2 = self._tick()
            t_rel += t2 - t1

            if rec is not None:
                px1, py1, px2, py2 = map(int, pbox_img)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), th)
                put(frame, 'person', (px1, max(py1 - 4, 12)), fs, (0, 255, 0), th)
                pc = ((px1 + px2) // 2, (py1 + py2) // 2)
                for j, b in enumerate(obox_img):
                    x1, y1, x2, y2 = map(int, b)
                    c = col(int(rec['labels'][j]))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), c, th)
                    put(frame, f"{OBJ_CLASSES[int(rec['labels'][j])]} {rec['scores'][j]:.2f}", (x1, max(y1 - 4, 12)), fs, c, th)
                lines = []
                for s, j, name in rels:
                    x1, y1, x2, y2 = map(int, obox_img[j])
                    cv2.line(frame, pc, ((x1 + x2) // 2, (y1 + y2) // 2), (255, 255, 255), 1, cv2.LINE_AA)
                    trip = f"person - {name} - {OBJ_CLASSES[int(rec['labels'][j])]}"
                    seen[trip] += 1
                    lines.append(f'{trip}  ({s:.2f})')
                if lines:
                    bw = max(cv2.getTextSize(s_, FONT, fs, th)[0][0] for s_ in lines) + 16
                    frame[:lh * len(lines) + 10, :bw] = (frame[:lh * len(lines) + 10, :bw] * 0.4).astype(np.uint8)
                    for q, s_ in enumerate(lines):
                        put(frame, s_, (8, lh * (q + 1)), fs, (255, 255, 255), th)
            if rec is not None:
                for lab in rec['labels']:
                    obj_cnt[OBJ_CLASSES[int(lab)]] += 1
                for sc, j, name in rels_all:
                    key = f"person - {name} - {OBJ_CLASSES[int(rec['labels'][j])]}"
                    L = segs.setdefault(key, [])
                    if L and i - L[-1]['end'] <= gap:
                        L[-1]['end'] = i; L[-1]['n'] += 1; L[-1]['sum'] += sc
                    else:
                        L.append(dict(start=i, end=i, n=1, sum=sc))
            writer.write(frame)
            i += 1
            if progress and n_total and i % 10 == 0:
                progress(min(i / (min(n_total, int(max_sec * fps)) if max_sec else n_total), 1.0))
        cap.release()
        writer.release()
        if n_bad:
            print(f'[demo] đã bỏ {n_bad} detection có toạ độ NaN/inf (thử đặt HALF=0 để chạy fp32 nếu nhiều)')
        if i == 0:
            raise RuntimeError('Không xử lý được frame nào (video rỗng, hoặc max_sec quá nhỏ).')
        tot = time.perf_counter() - t_all
        info = dict(frames=i, seconds=round(tot, 1), fps=round(i / max(tot, 1e-6), 1), source_fps=round(fps, 1),
                    ms_yolo=round(t_yolo / max(i, 1) * 1000, 1), ms_sttran=round(t_rel / max(i, 1) * 1000, 1),
                    top_triplets=seen.most_common(10))
        min_n = max(2, round(0.3 * fps))               # bỏ đoạn ngắn hơn ~0.3 s (nhiễu)
        segments = sorted(
            (dict(triplet=k, start=round(v['start'] / fps, 2), end=round((v['end'] + 1) / fps, 2), frames=v['n'],
                  avg_score=round(v['sum'] / v['n'], 3)) for k, L in segs.items() for v in L if v['n'] >= min_n),
            key=lambda d: (d['start'], d['triplet']))
        info['timeline'] = dict(video=os.path.basename(video), fps=round(fps, 2), duration_sec=round(i / fps, 2),
                                params=dict(conf=conf, rel_thr=rel_thr, ctx_sec=ctx_sec),
                                objects_seen=dict(obj_cnt.most_common()), segments=segments)
        return info


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', default='demo_out.mp4')
    ap.add_argument('--yolo', default=os.environ.get('YOLO_W', 'weights/yolo_ag4_best.pt'))
    ap.add_argument('--stt', default=os.environ.get('STT_W', 'weights/sttran4_best.pth'))
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--ctx_sec', type=float, default=0.5)
    ap.add_argument('--rel_thr', type=float, default=0.30)
    ap.add_argument('--topk', type=int, default=4)
    ap.add_argument('--max_sec', type=float, default=0)
    ap.add_argument('--json', default='', help='ghi timeline (JSON cho LLM) ra file này')
    a = ap.parse_args()
    d = Demo(a.yolo, a.stt, imgsz=a.imgsz)
    res = d.run(a.video, a.out, a.conf, a.ctx_sec, a.rel_thr, a.topk, max_sec=a.max_sec)
    tl = res.pop('timeline')
    print(res)
    if a.json:
        import json
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump(tl, f, ensure_ascii=False, indent=1)
        print('Đã ghi', a.json)
