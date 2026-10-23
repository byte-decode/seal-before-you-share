# Seal Before You Share

Code for the paper **"Seal Before You Share: A Nearly Invisible Semi-Fragile Watermark for Proactive Detection, Localization and Source Tracing of Face Forgeries"** (Rohit Agarwal, ByteDecode). The paper has been submitted to the *Journal of Information Security and Applications*.

## What this does, in short

A face photo gets an invisible watermark **before** it is published. The watermark carries a 32-bit owner code.

- Ordinary processing keeps the watermark: JPEG compression, resizing, light blur, small colour changes.
- Content replacement removes it in the replaced region: face swaps, face splicing and AI inpainting.

A verifier can then answer three questions:

1. **Has the image been manipulated?** The tamper map shows the pixels where the mark is missing.
2. **Where was it manipulated?** The tamper map outlines the edited region.
3. **Who published the original?** The 32-bit owner code usually survives the forgery.

The registry stores each owner code together with the face identity of the published photo. A watermark copied onto someone else's face is rejected. Images without a watermark go to a passive deepfake detector.

## Repository contents

| File | What it does | Paper section |
|---|---|---|
| `notebooks/01_extract_faces.ipynb` | Crops faces from FaceForensics++ and Celeb-DF videos | 4 (passive detectors) |
| `notebooks/02_train_passive_detectors.ipynb` | Trains Xception and EfficientNet-B4 deepfake classifiers | 5.1 |
| `notebooks/03a_train_watermark_v1.ipynb` | First watermark version (ablation) | 5.11 |
| `notebooks/03b_train_watermark_v2.ipynb` | v2: diverse masks, code on manipulations, group norm (ablation) | 5.11 |
| `notebooks/03c_train_watermark_v21.ipynb` | **v2.1, the final model** (v2 + stronger perceptual loss) | 3, 5 |
| `notebooks/04_evaluation.ipynb` | Main evaluation: quality, robustness, detection, localization, tracing, baselines, attacks, hybrid, figures | 5.2–5.11 |
| `notebooks/05_additional_experiments.ipynb` | Fixed operating point, geometric search, identity-bound verification | 5.6, 5.8, 5.9 |
| `scripts/make_tables.py` | Builds every LaTeX results table directly from the result CSV files | all tables |

## How to reproduce

All notebooks were run on Kaggle with one P100 or T4 GPU. Each notebook begins with a short **Kaggle settings** cell that lists the inputs to attach. Run the notebooks in order:

1. **Notebook 01** runs on CPU. Inputs: the FaceForensics++ c23 and Celeb-DF v2 Kaggle datasets. Save its output as a dataset named `deepfake-faces`.
2. **Notebook 02** needs `deepfake-faces` and produces `det_xception.pt` and `det_effb4.pt`.
3. **Notebooks 03a/03b/03c** need CelebA-HQ (256×256). Run each one twice: once with `MODE = "fixed"` and once with `MODE = "adaptive"`. Each run produces `wm_<mode>[_v2|_v21].pt` and `celeba_split.csv`.
4. **Notebook 04** needs all `wm_*.pt` files, `celeba_split.csv`, the `det_*.pt` files and CelebA-HQ. Internet must be on: it downloads InsightFace, Stable Diffusion inpainting and the baseline watermark models through Meta's [Video Seal](https://github.com/facebookresearch/videoseal) repository. It writes `final_*.csv` and the figures.
5. **Notebook 05** uses the same inputs and writes `add_*.csv`.
6. Rebuild the tables:
   ```bash
   python scripts/make_tables.py <folder with final_*.csv> <folder with add_*.csv>
   ```

Main dependencies: PyTorch, torchvision, timm, lpips, torchmetrics, opencv-python, onnxruntime, insightface models (downloaded at run time), diffusers, scikit-learn, pandas. The notebooks install the missing packages themselves. See `requirements.txt` for the list.

## Datasets (not included; download from the original sources)

- CelebA-HQ: Karras et al., 2018
- FaceForensics++: Rössler et al., 2019
- Celeb-DF v2: Li et al., 2020

Each dataset has its own licence and terms of use, which you must follow.

## Third-party models

The evaluation downloads InsightFace (InSwapper-128, ArcFace), Stable Diffusion v1.5 inpainting and the public baseline weights (HiDDeN, MBRS, CIN, TrustMark, WAM, all via Video Seal). They are used for research evaluation only, under their own licences. InSwapper-128 is licensed for non-commercial research only.

## Responsible use

The face swaps and inpaintings here are generated only to test the detector. Please do not use this code to create or spread deceptive media.

## Citation

```bibtex
@article{agarwal2026seal,
  title   = {Seal Before You Share: A Nearly Invisible Semi-Fragile Watermark for Proactive Detection, Localization and Source Tracing of Face Forgeries},
  author  = {Agarwal, Rohit},
  journal = {Journal of Information Security and Applications},
  note    = {Under review},
  year    = {2026}
}
```

## Licence

The code in this repository is released under the MIT licence (see `LICENSE`). Datasets and third-party models keep their own licences.
