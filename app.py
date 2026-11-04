"""App Gradio: tải video lên -> video có box + triplet (người – quan hệ – vật).   Chạy:  python app.py"""
import json, os, shutil, subprocess, tempfile, traceback


def _ensure_ffmpeg():
    """Gradio chỉ tìm lệnh `ffmpeg` trong PATH. Máy Windows thường không có -> tải video lên (mov/avi/mkv...) bị lỗi.
    Dùng bản ffmpeg đi kèm imageio-ffmpeg: chép ra thư mục riêng với tên `ffmpeg` rồi thêm vào PATH."""
    if shutil.which('ffmpeg'):
        return
    try:
        import imageio_ffmpeg
        src = imageio_ffmpeg.get_ffmpeg_exe()
        d = os.path.join(tempfile.gettempdir(), 'agapp_ffmpeg')
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg')
        if not os.path.exists(dst):
            shutil.copy(src, dst)
            os.chmod(dst, 0o755)
        os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')
    except Exception as e:
        print('[ffmpeg] không chuẩn bị được ffmpeg:', e)


_ensure_ffmpeg()

import gradio as gr

if shutil.which('ffprobe') is None:
    # Gradio gọi `ffprobe` để kiểm tra video có phát được không, nhưng imageio-ffmpeg chỉ có ffmpeg, không có ffprobe
    # -> FFExecutableNotFoundError. Video đầu ra đã được demo_core ghi sẵn H.264 nên bỏ qua bước kiểm tra này.
    from gradio import processing_utils as _pu
    _pu.video_is_playable = lambda *a, **k: True

from demo_core import Demo, probe_video, find_ffmpeg

YOLO_W = os.environ.get('YOLO_W', 'weights/yolo_ag4_best.pt')
STT_W = os.environ.get('STT_W', 'weights/sttran4_best.pth')
IMGSZ = int(os.environ.get('IMGSZ', '640'))      # phải bằng imgsz lúc train YOLO
_demo = None


def get_demo():
    global _demo
    if _demo is None:
        _demo = Demo(YOLO_W, STT_W, imgsz=IMGSZ)
    return _demo


def prepare_upload(video):
    """Chuyển video vừa tải lên sang H.264 mp4 chuẩn (có thời lượng) để trình duyệt xem trước được.
    Lỗi thì giữ nguyên file gốc, xử lý vẫn chạy bình thường (OpenCV tự giải mã được nhiều codec hơn trình duyệt)."""
    if not video:
        return video
    exe = find_ffmpeg()
    if not exe:
        return video
    try:
        out = os.path.join(tempfile.mkdtemp(), 'input_h264.mp4')
        r = subprocess.run([exe, '-y', '-loglevel', 'error', '-i', video, '-an', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
                            '-movflags', '+faststart', out], capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(out):
            print('[upload] đã chuyển sang H.264:', out)
            return out
        print('[upload] ffmpeg không chuyển được, giữ file gốc:', r.stderr.strip()[:400])
    except Exception as e:
        print('[upload] lỗi khi chuyển video, giữ file gốc:', e)
    return video


def infer(video, conf, ctx_sec, rel_thr, topk, max_sec, progress=gr.Progress()):
    if not video:
        raise gr.Error('Chưa chọn video.')
    try:
        d = get_demo()
        out = os.path.join(tempfile.mkdtemp(), 'demo_out.mp4')
        info = d.run(video, out, conf=conf, ctx_sec=ctx_sec, rel_thr=rel_thr, topk=int(topk), max_sec=max_sec,
                     progress=lambda p: progress(p, desc='Đang xử lý'))
    except Exception as e:                       # in traceback đầy đủ ra cmd, hiện lỗi thật trên giao diện
        traceback.print_exc()
        raise gr.Error(f'{type(e).__name__}: {e}')
    txt = (f"{info['frames']} frame trong {info['seconds']} s -> {info['fps']} FPS xử lý (video nguồn {info['source_fps']} FPS)\n"
           f"YOLO {info['ms_yolo']} ms | STTran {info['ms_sttran']} ms mỗi frame\n\nTriplet xuất hiện nhiều nhất:\n" +
           '\n'.join(f'  {n}: {c} frame' for n, c in info['top_triplets']))
    pv = probe_video(out)
    print('[output]', pv)
    jp = os.path.join(os.path.dirname(out), 'timeline.json')
    with open(jp, 'w', encoding='utf-8') as f:
        json.dump(info['timeline'], f, ensure_ascii=False, indent=1)
    return out, txt + '\n\nFile kết quả: ' + pv, jp


with gr.Blocks(title='Quan hệ người–vật (Action Genome)') as app:
    gr.Markdown('# Quan hệ người–vật trong video\nYOLO26 phát hiện người và vật, STTran dự đoán quan hệ (attention / spatial / contacting).')
    with gr.Row():
        with gr.Column():
            vin = gr.Video(label='Video đầu vào')
            conf = gr.Slider(0.01, 0.6, value=0.25, step=0.01, label='Ngưỡng phát hiện (conf)')
            ctx = gr.Slider(0.2, 1.5, value=0.5, step=0.1, label='Khoảng cách frame ngữ cảnh (giây)')
            thr = gr.Slider(0.1, 0.8, value=0.30, step=0.05, label='Xác suất tối thiểu của quan hệ')
            topk = gr.Slider(1, 15, value=6, step=1, label='Số triplet hiện trên video')
            msec = gr.Slider(0, 120, value=0, step=5, label='Chỉ xử lý n giây đầu (0 = cả video)')
            btn = gr.Button('Chạy', variant='primary')
        with gr.Column():
            vout = gr.Video(label='Kết quả')
            info = gr.Textbox(label='Thông tin', lines=10)
            jout = gr.File(label='JSON cho LLM (timeline.json)')
    vin.upload(prepare_upload, vin, vin)
    btn.click(infer, [vin, conf, ctx, thr, topk, msec], [vout, info, jout])

if __name__ == '__main__':
    app.launch()
