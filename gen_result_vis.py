#!/usr/bin/env python3
"""
Generate side-by-side OCR result images for the 4 real-world test cases.
Output: assets/realworld_ocr/{name}_result.jpg  (annotated bounding boxes + text)
        assets/realworld_ocr/{name}_panel.jpg    (2-column: original | annotated)
"""
import json, sys, textwrap
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
IMAGES = {
    "business_card":    ROOT / "assets/realworld_ocr/business_card.jpg",
    "dot_matrix":       ROOT / "assets/realworld_ocr/dot_matrix.jpg",
    "tire_sidewall":    ROOT / "assets/realworld_ocr/tire_sidewall.jpg",
    "elevator_display": ROOT / "assets/realworld_ocr/elevator_display.jpg",
}
OUT_DIR = ROOT / "assets/realworld_ocr"

# Use Medium model for best quality
MODEL_ROOT = ROOT / "ppocrv6_medium_onnx"

DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
REC_H, REC_MAX_W = 48, 2400
DB_THRESH, BOX_THRESH, UNCLIP, MIN_SIZE, MAX_EDGE = 0.20, 0.40, 1.40, 3, 960

# Colors
BOX_COLOR  = (50, 205, 100)   # green boxes
TEXT_BG    = (0, 0, 0, 180)   # semi-transparent black
TEXT_COLOR = (255, 255, 255)
PANEL_BG   = (20, 20, 20)


def get_providers():
    avail = ort.get_available_providers()
    p = ["CoreMLExecutionProvider"] if "CoreMLExecutionProvider" in avail else []
    return p + ["CPUExecutionProvider"]


def load_models():
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 4; opts.intra_op_num_threads = 4
    prov = get_providers()
    det = ort.InferenceSession(str(MODEL_ROOT/"det"/"inference.onnx"), sess_options=opts, providers=prov)
    rec = ort.InferenceSession(str(MODEL_ROOT/"rec"/"inference.onnx"), sess_options=opts, providers=prov)
    # Load char dict from yml (medium model doesn't have a json dict)
    import yaml
    cfg = yaml.safe_load((MODEL_ROOT/"rec"/"inference.yml").read_text(encoding="utf-8"))
    d = cfg.get("PostProcess", {}).get("character_dict", [])
    if not d:  # fallback to tiny json
        d = json.loads((ROOT/"ppocrv6_onnx/rec/char_dict.json").read_text(encoding="utf-8"))
    return det, rec, [""] + d + [" "]


def det_preprocess(img):
    w, h = img.size
    ratio = min(1.0, MAX_EDGE / max(w, h))
    nw = max(32, round(w * ratio / 32) * 32)
    nh = max(32, round(h * ratio / 32) * 32)
    arr = (np.array(img.resize((nw, nh)).convert("RGB"), dtype=np.float32)/255.0 - DET_MEAN) / DET_STD
    return arr.transpose(2, 0, 1)[np.newaxis], w/nw, h/nh, nw, nh


def cc_boxes(prob, pw, ph, sx, sy):
    binary = (prob > DB_THRESH).astype(np.uint8)
    flat, lf = binary.ravel(), np.zeros(pw*ph, dtype=np.int32)
    boxes = []
    for s in range(pw*ph):
        if flat[s] != 1 or lf[s] != 0: continue
        stack = [s]; lf[s] = 1; xs, ys, vs = [], [], []
        while stack:
            p = stack.pop(); x, y = p%pw, p//pw
            xs.append(x); ys.append(y); vs.append(prob[y,x])
            for nx2,ny2 in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if 0<=nx2<pw and 0<=ny2<ph:
                    i2=ny2*pw+nx2
                    if flat[i2]==1 and lf[i2]==0: lf[i2]=1; stack.append(i2)
        bw=max(xs)-min(xs)+1; bh=max(ys)-min(ys)+1
        if min(bw,bh)<MIN_SIZE or float(np.mean(vs))<BOX_THRESH: continue
        d=bw*bh*UNCLIP/(2*(bw+bh))
        boxes.append({"x0":max(0,(min(xs)-d)*sx),"y0":max(0,(min(ys)-d)*sy),
                      "x1":(max(xs)+d)*sx,"y1":(max(ys)+d)*sy,
                      "cy":(min(ys)+max(ys))/2*sy})
    boxes.sort(key=lambda b:(round(b["cy"]/10)*10,b["x0"]))
    return boxes


def run_det(sess, img):
    t,sx,sy,pw,ph = det_preprocess(img)
    return cc_boxes(sess.run(None,{sess.get_inputs()[0].name:t})[0][0,0], pw,ph,sx,sy)


def ctc_decode(out, char_list):
    prev=-1; txt=""
    for idx in out.argmax(axis=1):
        if idx!=0 and idx!=prev and idx<len(char_list): txt+=char_list[idx]
        prev=idx
    return txt


def run_rec(sess, img, boxes, char_list):
    texts=[]
    for box in boxes:
        x0,y0=int(max(0,box["x0"])),int(max(0,box["y0"]))
        x1,y1=min(int(box["x1"]),img.width),min(int(box["y1"]),img.height)
        if x1<=x0 or y1<=y0: continue
        crop=img.crop((x0,y0,x1,y1)).convert("RGB")
        cw,ch=crop.size
        nw=max(8,min(REC_MAX_W,round(cw*REC_H/ch)))
        arr=(np.array(crop.resize((nw,REC_H)),dtype=np.float32)/255.0-0.5)/0.5
        t=arr.transpose(2,0,1)[np.newaxis]
        out=sess.run(None,{sess.get_inputs()[0].name:t})[0][0]
        txt=ctc_decode(out,char_list)
        if txt.strip(): texts.append(txt)
        else: texts.append("")
    return texts


def try_font(size):
    for name in [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]:
        try: return ImageFont.truetype(name, size)
        except: pass
    return ImageFont.load_default()


def make_annotated(img: Image.Image, boxes, texts) -> Image.Image:
    """Draw colored boxes + text labels on a copy of the image."""
    vis = img.convert("RGBA").copy()
    overlay = Image.new("RGBA", vis.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    font_sm = try_font(13)

    for i, box in enumerate(boxes):
        x0,y0,x1,y1 = int(box["x0"]),int(box["y0"]),int(box["x1"]),int(box["y1"])
        draw.rectangle([x0,y0,x1,y1], outline=BOX_COLOR+(220,), width=2)
        txt = texts[i] if i < len(texts) else ""
        if txt:
            tw = min(len(txt)*8, x1-x0+20)
            ty = max(0, y0-18)
            draw.rectangle([x0, ty, x0+tw, ty+16], fill=TEXT_BG)
            draw.text((x0+2, ty+1), txt[:30], fill=TEXT_COLOR, font=font_sm)

    vis = Image.alpha_composite(vis, overlay).convert("RGB")
    return vis


def make_panel(orig: Image.Image, annotated: Image.Image,
               texts: list, label: str) -> Image.Image:
    """
    3-row panel:
      Row 1 (label bar)
      Row 2 (orig | annotated side by side)
      Row 3 (recognized text block)
    """
    PAD = 16
    LABEL_H = 36
    TEXT_AREA_H = 160

    # Scale both images to same height, max 480px
    target_h = min(480, orig.height)
    def scale(im, h):
        ratio = h / im.height
        return im.resize((int(im.width*ratio), h), Image.LANCZOS)

    orig_s = scale(orig, target_h)
    ann_s  = scale(annotated, target_h)

    total_w = orig_s.width + ann_s.width + PAD*3
    total_h = LABEL_H + target_h + PAD*2 + TEXT_AREA_H

    panel = Image.new("RGB", (total_w, total_h), PANEL_BG)
    draw  = ImageDraw.Draw(panel)
    font_title = try_font(18)
    font_body  = try_font(14)

    # Label bar
    draw.rectangle([0,0,total_w,LABEL_H], fill=(35,35,35))
    draw.text((PAD, 8), f"PP-OCRv6 Medium · {label}", fill=(200,200,200), font=font_title)

    # Images
    panel.paste(orig_s,  (PAD, LABEL_H + PAD))
    panel.paste(ann_s,   (PAD*2 + orig_s.width, LABEL_H + PAD))

    # Sub-labels
    draw.text((PAD+4, LABEL_H+PAD+target_h+4), "原图", fill=(150,150,150), font=font_body)
    draw.text((PAD*2+orig_s.width+4, LABEL_H+PAD+target_h+4), "检测框", fill=(50,205,100), font=font_body)

    # Text area
    ty = LABEL_H + PAD + target_h + 24
    draw.rectangle([0, ty-4, total_w, total_h], fill=(28,28,28))
    recognized = " / ".join(t for t in texts if t.strip())
    wrapped = textwrap.fill(recognized, width=max(40, total_w//10))
    draw.text((PAD, ty+4), wrapped[:400], fill=(220,220,200), font=font_body)

    return panel


def run(name, img_path, pre_rotate=0):
    print(f"\n{'='*50}")
    print(f"Processing: {name}")
    img_orig = Image.open(img_path).convert("RGB")

    # For slanted images, rotate before detection so text is roughly horizontal
    if pre_rotate:
        img_work = img_orig.rotate(pre_rotate, expand=True,
                                   fillcolor=(128, 128, 128))
    else:
        img_work = img_orig

    boxes = run_det(det, img_work)
    texts = run_rec(rec, img_work, boxes, char_list)
    print(f"  Detected {len(boxes)} boxes")
    print(f"  Text: {' | '.join(t for t in texts if t)[:120]}")

    annotated = make_annotated(img_work, boxes, texts)
    # Panel: show original on left, annotated (possibly rotated) on right
    panel = make_panel(img_orig, annotated, texts, name.replace("_"," ").title())

    ann_path = OUT_DIR / f"{name}_annotated.jpg"
    panel_path = OUT_DIR / f"{name}_panel.jpg"
    annotated.save(ann_path, quality=90)
    panel.save(panel_path, quality=90)
    print(f"  Saved: {ann_path.name}, {panel_path.name}")
    return texts


if __name__ == "__main__":
    print("Loading PP-OCRv6 Medium models...")
    det, rec, char_list = load_models()
    print(f"  Backend: {det.get_providers()[0]}")

    # tire sidewall text is at ~32° angle — pre-rotate to make it horizontal
    ROTATIONS = {"tire_sidewall": 32}
    for name, path in IMAGES.items():
        run(name, path, pre_rotate=ROTATIONS.get(name, 0))

    print("\nDone. Check assets/realworld_ocr/")
