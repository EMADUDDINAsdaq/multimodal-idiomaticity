# Multimodal Idiomaticity Representation

Given a context sentence containing a potentially idiomatic nominal compound (e.g. "bad apple") and three candidate images, predict which image matches the sense in which the compound is used in that sentence. Built for coursework at Newcastle University (MSc Data Science & AI — Advanced AI module, CSC8645).

**Why:** the same phrase can be literal or idiomatic depending on context — "a bad apple" spoiling a fruit bowl versus "a bad apple" in an organisation. Language models often struggle with this kind of figurative meaning; the task is designed to push toward representations that capture it, using paired text-and-image evidence rather than text alone.

Full write-up: [`report/Task2_Report.pdf`](report/Task2_Report.pdf).

## Dataset

Coursework-provided, via three splits:

| Split | Samples |
|---|---|
| Train | 231 |
| Validation | 27 |
| Test | 27 |

77 unique idiomatic compounds, each appearing exactly 3 times per split (1 context sentence, 3 candidate images). Columns: `compound`, `sentence_type`, `sentence`, `image_name`, `image_caption`, `label` (1 = correct image, 0 = incorrect). No null values across any split. Sentence length averaged 22 words (std 5.9, range 11–37). The label split was 154 negatives to 77 positives — a 2:1 imbalance that set the `pos_weight` used in every trained model.

**Access:** this assignment is built on **SemEval-2025 Task 1 — AdMIRe (Advancing Multimodal Idiomaticity Representation)** [3]. The task's labelled training data is openly hosted by the University of Sheffield on its ORDA/Figshare repository, [here](https://orda.shef.ac.uk/articles/dataset/AdMIRe_Advancing_Multimodal_Idiomaticity_Representation_SemEval-2025_Task_1_-_Labelled_Datasets/28436600), which states the data "can be shared openly." The column structure matches exactly (compound, context sentence, candidate images with captions, label).

The copy used for this coursework was distributed via the university's internal course platform rather than downloaded directly from ORDA, and may be a fixed subset chosen for the assignment rather than the full public release — so it's *not included in this repo* as-is. To reproduce this project from a fully public source, download the English training data from the ORDA link above instead; the notebook expects `train/`, `val/`, and `test/` folders, each containing a `.csv` with the columns above and an `images/` subfolder.

## Approach

**1. Data augmentation.** `Vamsi/T5_Paraphrase_Paws` [1] generated up to two paraphrases per training sentence (top-k = 50, top-p = 0.9, temperature = 1.0); outputs under 4 words or identical to the original were discarded. This produced 459 new rows, taking the training set from 231 to **690 samples** (461 negative, 229 positive, `pos_weight` = 2.01). All non-sentence fields (compound, image, label) were carried over unchanged — the rationale being that CLIP encodes meaning rather than exact wording, so a paraphrase should land close to the original sentence in embedding space.

**2. Feature extraction.** CLIP ViT-B/32 [2] encoded each image to a 512-d vector and each text (compound + sentence) to a 512-d vector. Both were L2-normalised and concatenated into a shared **1024-d joint embedding** — the common input to every model below. Feature extraction ran once and the resulting tensors were reused across all training runs.

**3. Four prediction approaches**, compared head-to-head:

| # | Approach | Description |
|---|---|---|
| — | **Zero-shot** | Raw CLIP cosine similarity between text and each candidate image, no training; highest-scoring image in each group of 3 wins. |
| 1 | **Similarity MLP** | 3 hand-crafted features (cosine similarity, image L2-norm, text L2-norm) → `Linear(3→16)` + ReLU + `Linear(16→1)`. `BCEWithLogitsLoss` (pos_weight=2.0), Adam lr=0.001, 20 epochs. |
| 2 | **Full-Embedding MLP (231)** | Full 1024-d CLIP vector → `Linear(1024→128)` + ReLU + `Linear(128→1)`. Same loss/optimiser as Model 1, trained on the original 231 samples, 20 epochs. |
| 3 | **Full-Embedding MLP (690)** | Same architecture as Model 2, trained on the 690 augmented samples, 15 epochs, pos_weight=2.01. |

All four use groupwise prediction at inference: within each group of 3 candidate images for a sentence, the highest-scoring (sigmoid) image is predicted correct.

## Results

| Approach | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Zero-shot — CLIP similarity | 0.33 | 0.00 | 0.00 | 0.00 |
| Model 1 — Similarity MLP | 0.63 | 0.44 | 0.44 | 0.44 |
| Model 2 — Full-Embedding MLP (231) | 0.93 | 0.89 | 0.89 | 0.89 |
| Model 3 — Full-Embedding MLP (690, augmented) | 0.93 | 0.89 | 0.89 | 0.89 |

Zero-shot CLIP similarity performs at chance (0.33 accuracy, F1 0.00 — it never predicts the correct image above chance), confirming that raw similarity alone can't resolve idiomatic meaning. The largest gain comes from Model 1 → Model 2: swapping 3 summary features for the full 1024-d embedding takes accuracy from 0.63 to 0.93. Model 3 shows paraphrase augmentation holds that same performance without regression, rather than improving on it further — consistent with CLIP encoding meaning rather than exact wording, so the extra paraphrased rows add training signal without adding new information in embedding space.

## Repository structure

```
multimodal-idiomaticity/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── Task2_Multimodal_Idiomaticity.ipynb
└── report/
    └── Task2_Report.pdf
```

## Running the notebook

Developed and run on Google Colab. The notebook mounts Google Drive and copies a zipped dataset from a personal folder (`drive.mount(...)`, then `!cp .../Task 2 NLP Dataset.zip`). To run it elsewhere:

1. Remove or skip the `google.colab` import and `drive.mount(...)` cell.
2. Obtain the dataset (see **Dataset** above for expected structure) and set `fpath` to point at it.
3. Install dependencies: `pip install -r requirements.txt` — note the CLIP install is from GitHub directly (`git+https://github.com/openai/CLIP.git`), not PyPI.
4. Run cells top to bottom — EDA, augmentation, CLIP feature extraction, then each of the four approaches in turn.

## References

[1] C. Raffel et al., "Exploring the limits of transfer learning with a unified text-to-text transformer," *J. Mach. Learn. Res.*, vol. 21, no. 140, pp. 1–67, 2020.
[2] A. Radford et al., "Learning transferable visual models from natural language supervision," *Proc. ICML*, 2021.
[3] T. Pickard, A. Villavicencio, M. Mi, W. He, D. Phelps, M. Idiart, "SemEval-2025 Task 1: AdMIRe — Advancing Multimodal Idiomaticity Representation," *Proc. 19th Int. Workshop on Semantic Evaluation (SemEval-2025)*, 2025.
