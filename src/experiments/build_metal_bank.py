import os, sys
ROOT = r"D:\programme file\ResearchArticle"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from PIL import Image
import numpy as np
from perception.winclip_patch import WinCLIPPatch

print("加载CLIP...")
det = WinCLIPPatch("ViT-B-16")
good_dir = os.path.join(ROOT, "data", "mvtec", "metal_nut", "train", "good")
files = sorted([f for f in os.listdir(good_dir) if f.lower().endswith((".png",".jpg"))])[:80]
feats = []
for i, f in enumerate(files):
    img = Image.open(os.path.join(good_dir, f)).convert("RGB")
    feats.append(det._get_patch_feats(img))
    if (i+1) % 20 == 0:
        print(f"  {i+1}/{len(files)}")
det.bank = np.concatenate(feats, axis=0)
out = os.path.join(ROOT, "results", "perception", "banks", "金属螺母(metal_nut).npy")
os.makedirs(os.path.dirname(out), exist_ok=True)
np.save(out, det.bank)
print(f"✅ {out}  shape={det.bank.shape}")
