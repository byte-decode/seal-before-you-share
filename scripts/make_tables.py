"""Builds every results table of the journal paper directly from the Kaggle result files (no hand-copied numbers)."""
import os, sys
import numpy as np
import pandas as pd

R = sys.argv[1]            # folder with final_*.csv (notebook 8)
A = sys.argv[2] if len(sys.argv) > 2 else None   # folder with add_*.csv (notebook 9), optional
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
os.makedirs(OUT, exist_ok=True)
rd = lambda f: pd.read_csv(os.path.join(R, f), keep_default_na=False)

MAIN = "Ours v2.1 (fixed)"
BASE = ["HiDDeN (matched PSNR)", "MBRS (matched PSNR)", "CIN (matched PSNR)", "TrustMark (matched PSNR)", "WAM (matched PSNR)"]
SHORT = {MAIN: "\\textbf{Ours}", "Ours v2.1 (adaptive)": "Ours (adaptive)",
         **{b: b.replace(" (matched PSNR)", "") for b in BASE}}
COMPARE = [MAIN] + BASE

def best_mark(vals, higher=True, tol=1e-9):
    arr = np.array(vals, float)
    b = np.nanmax(arr) if higher else np.nanmin(arr)
    return [abs(v - b) <= tol for v in arr]

def write(name, body):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(body)

# ---------------- quality ----------------
q = rd("final_quality.csv").set_index("Method")
rows = []
lp = [q.loc[m, "LPIPS"] for m in COMPARE]
bl = best_mark(lp, higher=False)
for m, b in zip(COMPARE, bl):
    r = q.loc[m]
    lps = f"{r['LPIPS']:.4f}"
    rows.append(f"{SHORT[m]} & {int(r['Bits'])} & {r['PSNR (dB)']:.2f} & {r['SSIM']:.3f} & {'\\textbf{'+lps+'}' if b else lps} & {r['Embed (ms)']:.1f} & {r['Verify (ms)']:.1f} \\\\")
write("quality.tex", "\n".join(rows) + "\n")

# ---------------- robustness (bit accuracy %) ----------------
rb = rd("final_robustness.csv").set_index("Method")
cols = ["None", "JPEG 90", "JPEG 70", "JPEG 50", "JPEG 30", "Resize 0.75", "Resize 0.5", "Blur (sigma 1)",
        "Noise (sigma 2.5)", "Brightness +10%", "Social media (resize 0.75 + JPEG 75)"]
rows = []
for c, lab in zip(cols, ["None", "JPEG 90", "JPEG 70", "JPEG 50", "JPEG 30", "Resize 0.75", "Resize 0.5",
                         "Blur ($\\sigma{=}1$)", "Noise ($\\sigma{=}2.5$)", "Brightness +10\\%", "Resize 0.75 + JPEG 75"]):
    v = [100 * rb.loc[m, c] for m in COMPARE]
    bm = best_mark([round(x, 1) for x in v])
    cells = [("\\textbf{%.1f}" % x) if b else ("%.1f" % x) for x, b in zip(v, bm)]
    rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
mean = [100 * rb.loc[m, cols].astype(float).mean() for m in COMPARE]
bm = best_mark([round(x, 1) for x in mean])
rows.append("\\midrule\nMean & " + " & ".join([("\\textbf{%.1f}" % x) if b else ("%.1f" % x) for x, b in zip(mean, bm)]) + " \\\\")
write("robustness.tex", "\n".join(rows) + "\n")

# ---------------- detection: TPR@1%FPR and AUC ----------------
d = rd("final_detection.csv")
man = ["Face swap", "Face splice", "Diffusion inpainting"]
rows = []
for m in COMPARE:
    sub = d[d.Method == m]
    score = "tamper map" if "tamper map" in set(sub.Score) else "code errors"
    s = sub[sub.Score == score]
    cells = []
    for mn in man:
        for post in ["raw", "mixed edits"]:
            r = s[(s.Manipulation == mn) & (s.Post == post)].iloc[0]
            cells.append(f"{r['AUC']:.3f} / {100 * r['TPR@1%FPR']:.1f}")
    lab = SHORT[m] + (" (map)" if score == "tamper map" else " (code)")
    rows.append(lab + " & " + " & ".join(cells) + " \\\\")
write("detection.tex", "\n".join(rows) + "\n")

# ---------------- localization IoU + code survival after manipulation ----------------
l = rd("final_localization.csv")
rows = []
for m in ["Ours v2.1 (fixed)", "Ours v2.1 (adaptive)", "WAM (matched PSNR)"]:
    cells = []
    for mn in man:
        for post in ["raw", "JPEG 75"]:
            r = l[(l.Method == m) & (l.Manipulation == mn) & (l.Post == post)].iloc[0]
            cells.append(f"{r['IoU']:.3f}")
    rows.append(SHORT[m] + " & " + " & ".join(cells) + " \\\\")
write("localization.tex", "\n".join(rows) + "\n")

rows = []
for m in COMPARE:
    s = d[(d.Method == m) & (d.Score == "code errors")]
    cells = []
    for mn in man:
        for post in ["raw", "mixed edits"]:
            cells.append(f"{100 * s[(s.Manipulation == mn) & (s.Post == post)].iloc[0]['Bit acc. on fakes']:.1f}")
    rows.append(SHORT[m] + " & " + " & ".join(cells) + " \\\\")
write("tracing.tex", "\n".join(rows) + "\n")

# ---------------- regeneration + copy attacks (code only) ----------------
a = rd("final_attacks.csv")
att = ["AI regeneration (VAE)", "AI regeneration (diffusion, strength 0.15)",
       "Copy attack (exact watermark pasted on another face)", "Copy attack (estimated watermark pasted on another face)"]
rows = []
for m in COMPARE:
    cells = []
    for an in att:
        r = a[(a.Method == m) & (a.Attack == an)].iloc[0]
        cells.append(f"{100 * float(r['Bit accuracy']):.1f} / {float(r['Owner code verified (%)']):.1f}")
    rows.append(SHORT[m] + " & " + " & ".join(cells) + " \\\\")
write("attacks.tex", "\n".join(rows) + "\n")

# regeneration flagged by tamper map at the 1%-of-pixels rule, from final_decisions (threshold there was degenerate) -> use flagged area mean
rows = []
for m in ["Ours v2.1 (fixed)", "Ours v2.1 (adaptive)", "WAM (matched PSNR)"]:
    cells = []
    for an in att[:2]:
        r = a[(a.Method == m) & (a.Attack == an)].iloc[0]
        cells.append(f"{100 * float(r['Flagged area (mean)']):.1f}")
    rows.append(SHORT[m] + " & " + " & ".join(cells) + " \\\\")
write("regen_flag.tex", "\n".join(rows) + "\n")

# ---------------- ablation ----------------
abl = [("Ours v1 (fixed)", "v1", "fixed"), ("Ours v1 (adaptive)", "v1", "adaptive"),
       ("Ours v2 (fixed)", "v2", "fixed"), ("Ours v2 (adaptive)", "v2", "adaptive"),
       ("Ours v2.1 (fixed)", "v2.1", "fixed"), ("Ours v2.1 (adaptive)", "v2.1", "adaptive")]
rows = []
for m, v, s in abl:
    lpips_ = q.loc[m, "LPIPS"]; j30 = 100 * rb.loc[m, "JPEG 30"]
    loc = l[(l.Method == m) & (l.Post == "JPEG 75")].set_index("Manipulation")["IoU"]
    det = d[(d.Method == m) & (d.Score == "code errors") & (d.Post == "mixed edits")].set_index("Manipulation")["Bit acc. on fakes"]
    tpr = d[(d.Method == m) & (d.Score == "tamper map") & (d.Post == "mixed edits") & (d.Manipulation == "Diffusion inpainting")]["TPR@1%FPR"].iloc[0]
    vae = a[(a.Method == m) & (a.Attack == att[0])]["Owner code verified (%)"].astype(float).iloc[0]
    rows.append(f"{v} & {s} & {lpips_:.4f} & {j30:.1f} & {loc['Face swap']:.3f} & {loc['Diffusion inpainting']:.3f} & "
                f"{100 * det['Face swap']:.1f} & {100 * tpr:.1f} & {vae:.1f} \\\\")
write("ablation.tex", "\n".join(rows) + "\n")

# ---------------- hybrid ----------------
h = rd("final_hybrid.csv")
rows = []
for wm in ["Ours v2.1 (fixed)", "WAM (matched PSNR)"]:
    for sysname in ["Passive only (EfficientNet-B4)", "Passive only (Xception)", "Watermark only", "Hybrid (EfficientNet-B4)", "Hybrid (Xception)"]:
        r = h[(h.Watermark == wm) & (h.System == sysname)].iloc[0]
        rows.append(f"{SHORT[wm]} & {sysname.replace('EfficientNet-B4', 'EffNet-B4')} & {r['ACC']:.3f} & {r['Precision']:.3f} & {r['Recall']:.3f} & {r['F1']:.3f} & {r['ACC marked']:.3f} & {r['ACC unmarked']:.3f} \\\\")
    rows.append("\\midrule")
write("hybrid.tex", "\n".join(rows[:-1]) + "\n")

print("tables written to", OUT, sorted(os.listdir(OUT)))

# ======================= notebook 9 (additional experiments) =======================
PEND = "\\multicolumn{%d}{c}{\\textit{pending: results of notebook 9}} \\\\\n"
if A is None:
    write("op.tex", PEND % 4); write("geometric.tex", PEND % 8); write("identity.tex", PEND % 5)
else:
    ra = lambda f: pd.read_csv(os.path.join(A, f), keep_default_na=False)
    op = ra("add_operating_point.csv").set_index("Setting")
    mcols = [c for c in ["Ours v2.1 (fixed)", "Ours v2.1 (adaptive)", "WAM (matched PSNR)"] if c in op.columns]
    rows = []
    for s in op.index:
        if s == "Threshold (flagged area)":
            continue
        lab = s.replace("False alarm: ", "").replace("Detected: ", "").replace("Social media (resize 0.75 + JPEG 75)", "Resize 0.75 + JPEG 75").replace("sigma ", "$\\sigma{=}$").replace("%", "\\%")
        rows.append(lab + " & " + " & ".join(f"{float(op.loc[s, c]):.1f}" for c in mcols) + " \\\\")
    nfa = sum(1 for x in op.index if x.startswith("False alarm"))
    rows.insert(0, "\\multicolumn{4}{l}{\\textit{False alarms on genuine images}} \\\\")
    rows.insert(nfa + 1, "\\midrule\n\\multicolumn{4}{l}{\\textit{Detection of manipulated images}} \\\\")
    write("op.tex", "\n".join(rows) + "\n")
    g = ra("add_geometric.csv")
    rows = []
    for an in list(dict.fromkeys(g.Attack)):
        cells = []
        for m in COMPARE:
            r = g[(g.Method == m) & (g.Attack == an)].iloc[0]
            cells.append(f"{float(r['Verified % (no search)']):.0f} / {float(r['Verified % (search)']):.0f}")
        r = g[(g.Method == MAIN) & (g.Attack == an)].iloc[0]
        cells.append(f"{float(r['False alarm % (search)']):.0f}")
        rows.append(an.replace('°', '$^\\circ$').replace('%', '\\%').replace(' (off grid)', '$^{\\dagger}$') + " & " + " & ".join(cells) + " \\\\")
    fm = [f"{float(g[g.Method == m]['Unmarked faces falsely verified % (search)'].iloc[0]):.0f}" for m in COMPARE]
    rows.append("\\midrule\nUnmarked faces falsely verified & " + " & ".join(fm) + " & -- \\\\")
    write("geometric.tex", "\n".join(rows) + "\n")
    idt = ra("add_identity.csv")
    cases = list(dict.fromkeys(idt.Case))
    rows = []
    for m in COMPARE:
        cells = []
        for c in cases:
            r = idt[(idt.Method == m) & (idt.Case == c)].iloc[0]
            cells.append(f"{float(r['Accepted as genuine: code only (%)']):.1f} / {float(r['Accepted as genuine: code + identity (%)']):.1f}")
        rows.append(SHORT[m] + " & " + " & ".join(cells) + " \\\\")
    write("identity.tex", "\n".join(rows) + "\n")
print("done")
