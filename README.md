# Grad-CAM study on PASCAL VOC 2007 (EE798R Pattern Recognition course project)

Independent PyTorch implementation of Grad-CAM (Selvaraju et al., ICCV 2017) with vanilla saliency, CAM, Grad-CAM++
and a centre-Gaussian control, evaluated for object localisation on VOC2007 test. The LaTeX report is `main.tex`.

All numbers in the report come from the files in `results/quantitative/` (one full run on a Colab T4, seed 0), produced by the
scripts below and read into the LaTeX by `\input`. Re-running the scripts regenerates them.

## Layout
```
main.tex, references.bib       report (LaTeX; compile with latexmk -pdf main.tex)
src/gradcam.py                 Grad-CAM (+ Grad-CAM++) from scratch: forward hook + torch.autograd.grad
src/saliency.py, src/cam.py    baselines
src/dataset.py, src/model.py   VOC2007 wrapper (boxes + labels), ResNet-50 with 20-way head
src/evaluation.py              heat-map -> box, IoU, pointing game, mass, discrimination index, bootstrap
scripts/train.py               fine-tune layer4 + head on VOC07 trainval (90/10 split)
scripts/evaluate.py            all localisation metrics, threshold sweep, layer ablation -> results/quantitative/*.csv
scripts/benchmark.py           timing / memory / parameter count -> results/quantitative/timing.json
scripts/make_tables.py         CSV/JSON -> results/quantitative/*.tex tables + figures/threshold_sweep.png
scripts/generate_heatmaps.py   rule-based qualitative figures -> figures/fig_*.png
scripts/extra_analysis.py      breakdowns by object size / classifier confidence / class, from per_pair.csv -> results/quantitative/*.tex + extra.json
scripts/extract_snippets.py    copies marked code regions to snippets/ (quoted verbatim in the report)
tests/test_gradcam.py          correctness tests (random weights; produce no results)
```

## Environment
Python 3.10+ (developed with 3.12), PyTorch >= 2.1, torchvision >= 0.16. A GPU is strongly recommended (Colab T4 is enough).
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # on Colab: only `pip install scikit-learn pandas` is missing
```

## Data
VOC2007 trainval + test are downloaded by torchvision on first use (`--root data`; ~900 MB, host: Oxford VOC server).
If the server is down, download `VOCtrainval_06-Nov-2007.tar` and `VOCtest_06-Nov-2007.tar` manually, extract both into
`data/` so that `data/VOCdevkit/VOC2007/{JPEGImages,Annotations,ImageSets}` exists, and add `--no-download`.
Preprocessing is inside `src/dataset.py` (resize to 448x448, ImageNet normalisation, boxes rescaled, `difficult` objects ignored).

## Reproduce everything (in order)
```
python tests/test_gradcam.py                       # 1. verify the implementation (CAM identity etc.)
python scripts/train.py --root data                # 2. fine-tune (seed 0); writes checkpoints/ + train_log.json
python scripts/evaluate.py --root data             # 3. all metrics, sweeps, ablation, per-pair CSV, test probabilities
python scripts/benchmark.py                        # 4. timing/memory/params (run on the same machine; GPU if available)
python scripts/make_tables.py                      # 5. LaTeX tables + threshold figure from the measured files
python scripts/generate_heatmaps.py --root data    # 6. qualitative figures
python scripts/compress_figures.py                   # 6b. (optional) JPEG copies of the qualitative figures so the PDF is small
python scripts/extra_analysis.py                    # 7. size/confidence/class breakdowns used in the failure analysis
python scripts/extract_snippets.py                 # 8. refresh code snippets used in the report
latexmk -pdf main.tex                              # 9. build the report
```
Quick debug run (minutes, NOT for reporting): add `--max-train 200 --epochs 1` to train and `--max-images 100` to evaluate.
Colab: upload the folder, `!pip install scikit-learn pandas`, then run the lines above with `!` prefixes (Runtime -> GPU).

## Key settings (also in the report)
ResNet-50 (ImageNet weights), 448x448 input, AdamW (head 1e-3, layer4 1e-4, wd 1e-4), cosine schedule, batch 32, 6 epochs,
seed 0. Localisation threshold 0.15 of the map maximum (swept 0.05-0.60), largest connected component, IoU >= 0.5.
Target layers for the ablation: `layer2`, `layer3`, `layer4` (strides 8/16/32).

## Qualitative-example selection rules (no hand-picking; `--seed 0`)
success: IoU>=0.5 and pointing hit; partial: pointing hit and predicted-box area < 0.5 x GT area;
background: pointing miss and mass-in-GT < 0.3; misclassification: top-1 class absent from the image (explains the wrong class);
multi-object: images with >= 2 classes. Chosen ids are saved in `results/qualitative/selected_examples.json`.

## Code provenance / integrity
Libraries: PyTorch + torchvision (autodiff, ResNet-50, VOC parsing, ImageNet weights), SciPy (connected components),
scikit-learn (average precision), NumPy, pandas, Matplotlib. No external Grad-CAM code is used.
Everything else was written for this project. You are responsible for understanding it: read `src/gradcam.py` and
`tests/test_gradcam.py` before submitting, and follow your course's rules on disclosing AI assistance.

## Known limitations of the protocol
Single architecture/dataset; box (not mask) ground truth; no deletion/insertion faithfulness test; no occlusion/noise study;
FLOPs not measured. Grad-CAM can return an all-zero map when no location supports the class; these are counted as failures
and reported ("Empty maps" column).
