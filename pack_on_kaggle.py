# Dán vào MỘT cell cuối notebook Kaggle (phiên có runs/yolo/ag4 và runs/sttran4), chạy xong tải app_pack.zip ở panel Output.
import os, glob, shutil
os.chdir('/kaggle/working')
P = 'app_pack'
shutil.rmtree(P, ignore_errors=True); os.makedirs(f'{P}/weights')

def first(*pats):
    for p in pats:
        g = glob.glob(p)
        if g: return g[0]

Y = first('runs/yolo/ag4/weights/best.pt', 'backup/yolo__best.pt', '/kaggle/input/datasets/*/ag-ckpt-backup-v8/yolo__best.pt')
S = first('runs/sttran4/best.pth', 'backup/sttran4__best.pth', '/kaggle/input/datasets/*/ag-ckpt-backup-v8/sttran4__best.pth')
print('YOLO  :', Y); print('STTran:', S)
assert Y and S, 'Chưa thấy trọng số ag4 / sttran4'
shutil.copy(Y, f'{P}/weights/yolo_ag4_best.pt'); shutil.copy(S, f'{P}/weights/sttran4_best.pth')
for d in ('agyolo', 'agsgg'):                                   # code đã dùng để train (chỉ file .py)
    shutil.copytree(d, f'{P}/{d}', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'tests', 'tools'))
shutil.make_archive('app_pack', 'zip', P)
print('xong: /kaggle/working/app_pack.zip', round(os.path.getsize('app_pack.zip') / 1e6), 'MB')
