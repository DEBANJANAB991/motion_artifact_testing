#!/usr/bin/env python3
import argparse, re, csv
from pathlib import Path

FINAL_LINE = re.compile(
    r"Final\s+Test\s*[—-]?\s*Loss:\s*([0-9.]+)"
    r"(?:.*?\bPSNR:\s*([0-9.]+)\s*dB)?"
    r"(?:.*?\bSSIM:\s*([0-9.]+))?",
    re.IGNORECASE | re.DOTALL,
)

PARAMS_LINE = re.compile(r"Model\s+parameters:\s*([\d,]+)", re.IGNORECASE)

MODEL_FROM_NAME = {
    "mr_lkv": "MR_LKV",
    "mr_lkvv2": "MR_LKV",
    "replk": "RepLKNet",
    "replknet": "RepLKNet",
    "swinir": "SwinIR",
    "restormer": "Restormer",
    "unet": "UNet",
    "unet_model": "UNet",
}

def guess_model_name(p: Path, text: str) -> str:
    name = p.stem.lower()
    for key, pretty in MODEL_FROM_NAME.items():
        if key in name:
            return pretty
    m = re.search(r"Starting training with model\s*=\s*([A-Za-z0-9_]+)", text, re.IGNORECASE)
    if m:
        return MODEL_FROM_NAME.get(m.group(1).lower(), m.group(1))
    return p.stem

def parse_one(path: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    model = guess_model_name(path, text)
    m_final = FINAL_LINE.search(text)
    if not m_final:
        return None
    loss = float(m_final.group(1))
    psnr = float(m_final.group(2)) if m_final.group(2) else None
    ssim = float(m_final.group(3)) if m_final.group(3) else None
    m_params = PARAMS_LINE.search(text)
    params = int(m_params.group(1).replace(",", "")) if m_params else None
    return {
        "model": model,
        "final_test_loss": loss,
        "final_test_psnr": psnr,
        "final_test_ssim": ssim,
        "params": params,
        "log_file": str(path),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=Path("results/logs"),
                    help="Directory containing *.out/*.err logs")
    ap.add_argument("--out-csv", type=Path, default=Path("../results/tables/metrics_from_logs.csv"),
                    help="Where to write the CSV (relative to current dir)")
    ap.add_argument("--fig-dir", type=Path, default=Path("../results/figures"),
                    help="Where to save comparison figure (optional)")
    args = ap.parse_args()

    log_dir = args.log_dir
    print(f"[info] Searching logs in: {log_dir.resolve()}")
    files = sorted(list(log_dir.glob("*.out")) + list(log_dir.glob("*.err")))
    if not files:
        print("[warn] No files found.")
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as f:
            csv.writer(f).writerow(["model","final_test_loss","final_test_psnr","final_test_ssim","params","log_file"])
        print(f"[done] Wrote 0 rows to {args.out_csv}")
        return

    rows = []
    for fp in files:
        row = parse_one(fp)
        if row:
            rows.append(row)
        else:
            print(f"[skip] No 'Final Test' line parsed in {fp.name}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","final_test_loss","final_test_psnr","final_test_ssim","params","log_file"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[done] Wrote {len(rows)} rows to {args.out_csv}")

    

if __name__ == "__main__":
    main()
