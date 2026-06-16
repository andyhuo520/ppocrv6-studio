#!/usr/bin/env python3
"""
Generate OmniDocBench predictions for Small and Medium models,
then run the OmniDocBench evaluation to get text_block Edit_dist scores.

Saves predictions as .md files (same format as ppocrv6_browser predictions).
"""
import json, time, sys, subprocess
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT       = Path(__file__).parent
IMAGES_DIR = ROOT / "OmniDocBench/demo_data/omnidocbench_demo/images"
GT_JSON    = ROOT / "OmniDocBench/demo_data/omnidocbench_demo/OmniDocBench_demo.json"
BENCH_DIR  = ROOT / "OmniDocBench"
PRED_BASE  = ROOT / "predictions"
RESULT_BASE = ROOT / "results"

DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
REC_H, REC_MAX_W = 48, 2400
DB_THRESH, BOX_THRESH, UNCLIP, MIN_SIZE, MAX_EDGE = 0.20, 0.40, 1.40, 3, 960


def get_providers():
    avail = ort.get_available_providers()
    p = ["CoreMLExecutionProvider"] if "CoreMLExecutionProvider" in avail else []
    return p + ["CPUExecutionProvider"]


def load_model(variant: str):
    suffix = f"_{variant}" if variant != "tiny" else ""
    root = ROOT / f"ppocrv6{suffix}_onnx"
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 4; opts.intra_op_num_threads = 4
    prov = get_providers()
    det = ort.InferenceSession(str(root/"det"/"inference.onnx"), sess_options=opts, providers=prov)
    rec = ort.InferenceSession(str(root/"rec"/"inference.onnx"), sess_options=opts, providers=prov)
    char_json = root / "rec" / "char_dict.json"
    d = json.loads(char_json.read_text(encoding="utf-8")) if char_json.exists() else []
    if not d:
        import yaml
        cfg = yaml.safe_load((root/"rec"/"inference.yml").read_text(encoding="utf-8"))
        d = cfg.get("PostProcess", {}).get("character_dict", [])
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
    lab = np.zeros((ph, pw), dtype=np.int32)
    flat, lf = binary.ravel(), lab.ravel()
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
    return "\n".join(texts)


def generate_predictions(variant: str):
    pred_dir = PRED_BASE / f"ppocrv6_{variant}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*55}")
    print(f"  Generating predictions: PP-OCRv6 {variant.upper()}")
    print(f"  → {pred_dir}")
    print(f"{'='*55}")

    det, rec, char_list = load_model(variant)
    backend = det.get_providers()[0]
    print(f"  Backend: {backend}")

    img_files = sorted(IMAGES_DIR.glob("*.jpg"))
    timings = []
    for img_path in img_files:
        t0 = time.perf_counter()
        img = Image.open(img_path).convert("RGB")
        boxes = run_det(det, img)
        pred_text = run_rec(rec, img, boxes, char_list)
        elapsed = (time.perf_counter()-t0)*1000
        timings.append(elapsed)

        # Save as .md file (same convention as browser predictions)
        out_path = pred_dir / (img_path.stem + ".md")
        out_path.write_text(pred_text, encoding="utf-8")
        print(f"  {img_path.name[:52]:52s}  {elapsed:.0f}ms  {len(boxes)}boxes")

    avg = sum(timings)/len(timings) if timings else 0
    print(f"\n  {len(timings)} images done, avg {avg:.0f}ms/img")
    return pred_dir


def run_evaluation(variant: str, pred_dir: Path):
    """Write config and run OmniDocBench evaluation."""
    cfg_path = ROOT / f"configs_ppocrv6_{variant}.yaml"
    result_dir = RESULT_BASE / f"ppocrv6_{variant}"
    result_dir.mkdir(parents=True, exist_ok=True)

    cfg = f"""end2end_eval:
  metrics:
    text_block:
      metric:
        - Edit_dist
    reading_order:
      metric:
        - Edit_dist
  dataset:
    dataset_name: md2md_dataset
    ground_truth:
      data_path: {ROOT}/OmniDocBench/demo_data/omnidocbench_demo/mds
      page_info: {GT_JSON}
    prediction:
      data_path: {pred_dir}
    match_method: quick_match
"""
    cfg_path.write_text(cfg)
    print(f"\n  Running OmniDocBench evaluation for {variant}...")

    try:
        r = subprocess.run(
            [sys.executable, "-m", "tools.eval", "--config", str(cfg_path),
             "--result_dir", str(BENCH_DIR/"result")],
            cwd=str(BENCH_DIR), capture_output=True, text=True, timeout=120
        )
        print(r.stdout[-2000:] if r.stdout else "(no stdout)")
        if r.stderr:
            print("STDERR:", r.stderr[-1000:])
        return r.returncode == 0
    except Exception as e:
        print(f"  Eval error: {e}")
        return False


def read_score(variant: str):
    """Read the text_block Edit_dist score from result files."""
    import glob
    patterns = [
        f"{BENCH_DIR}/result/ppocrv6_{variant}_*text_block_result.json",
        f"{BENCH_DIR}/result/*{variant}*text_block_result.json",
    ]
    for pat in patterns:
        files = glob.glob(pat)
        if files:
            data = json.loads(Path(files[0]).read_text())
            # try to extract score
            if isinstance(data, list):
                eds = [e.get("edit", 0) for e in data if isinstance(e, dict)]
                return round(sum(eds)/len(eds), 4) if eds else None
            elif isinstance(data, dict):
                return data.get("Edit_dist") or data.get("edit_dist")
    return None


if __name__ == "__main__":
    variants = sys.argv[1:] or ["small", "medium"]

    for v in variants:
        pred_dir = generate_predictions(v)
        ok = run_evaluation(v, pred_dir)
        if ok:
            score = read_score(v)
            print(f"\n  PP-OCRv6 {v}: text_block Edit_dist = {score}")
        else:
            print(f"\n  Evaluation failed for {v}")
