# Clustering Analysis of Wholesale Customers
**ML Capstone Project — Group 4**

Unsupervised customer segmentation using K-Means, Agglomerative Hierarchical Clustering, and DBSCAN, with and without PCA dimensionality reduction.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Pipeline](#pipeline)
- [Algorithms & Hyperparameter Tuning](#algorithms--hyperparameter-tuning)
- [Results Summary](#results-summary)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Key Findings](#key-findings)

---

## Project Overview

This project explores whether meaningful customer segments can be discovered from wholesale purchase data using unsupervised learning — without any ground-truth labels. Three clustering algorithms are benchmarked on the same preprocessed data, evaluated using three internal validity metrics, and compared across both original 6-D feature space and a 2-D PCA-reduced space.

**Research question:** *Can annual spending patterns across six product categories reliably segment wholesale customers, and how do different algorithms compare on this task?*

---

## Dataset

**Source:** [UCI Wholesale Customers Dataset](https://archive.ics.uci.edu/ml/datasets/wholesale+customers)

| Property | Value |
|---|---|
| Samples | 440 |
| Features used | 6 (continuous spending variables) |
| Features excluded | Channel, Region (used for post-hoc validation only) |
| Missing values | None |

**Features:** `fresh`, `milk`, `grocery`, `frozen`, `detergents_paper`, `delicassen`

All features exhibit heavy right skew (skewness 4–8) and coefficients of variation exceeding 1, necessitating log-transformation and standardisation before clustering.

The dataset is loaded directly from Google Drive in the notebook:
```python
url = 'https://drive.google.com/uc?id=1yDhLUJoB-Z69OL2Ad7tvm_0Gt6Dh2a5o'
data = pd.read_csv(url)
```

---

## Repository Structure

```
.
├── Untitled10-Copy1.ipynb      # Main analysis notebook
├── cluster_plots.png           # PCA scatter plots for all three algorithms
├── selection_plots.png         # Silhouette curves + k-distance graph
├── clustering_report_5_page_draft.docx   # Draft report
├── Capstone_Group4_Final_Report.docx     # Final 5-page report (with figures)
└── README.md
```

---

## Pipeline

```
Raw CSV
   │
   ▼
Column normalisation (lowercase names)
   │
   ▼
Drop Channel & Region  →  Feature matrix X (440 × 6)
   │
   ▼
log1p transformation   (tame skewness 4–8)
   │
   ▼
StandardScaler         (zero mean, unit variance)
   │
   ├──────────────────────────────────────┐
   ▼                                      ▼
Cluster on X_scaled (6-D)         PCA(n_components=2)
   │                               71.3% variance retained
   ▼                                      │
Evaluate metrics                          ▼
                                  Cluster on X_pca (2-D)
                                          │
                                          ▼
                                  Evaluate metrics + plot
```

---

## Algorithms & Hyperparameter Tuning

### K-Means
- Tuning range: `k ∈ {2, …, 10}`
- Settings: `n_init=20`, `random_state=42`
- Selection method: silhouette peak + elbow in inertia curve
- **Optimal:** `k = 2`

```python
def tune_kmeans(X_scaled, k_range, random_state=42):
    # Returns DataFrame of silhouette, DBI, CH scores per k
```

### Agglomerative Hierarchical Clustering
- Linkage: Ward (minimises within-cluster variance)
- Tuning range: `n_clusters ∈ {2, …, 10}`
- Selection method: silhouette score
- **Optimal:** `n_clusters = 2`

```python
def tune_agglomerative(X_scaled, cluster_range, linkage="ward"):
    # Returns DataFrame of silhouette, DBI, CH scores per n_clusters
```

### DBSCAN
- `min_samples = 5` (standard rule-of-thumb for moderate-dimensional data)
- ε selected via 5-NN distance graph (knee of sorted distance curve ≈ 0.7)
- **Optimal:** `eps = 0.7, min_samples = 5`

```python
def compute_k_distance(X_scaled, min_samples=5):
    # Plots sorted k-NN distances; returns distances array

def tune_dbscan(X_scaled, param_grid):
    # Returns DataFrame of cluster count, noise %, silhouette, DBI, CH per (eps, min_samples)
```

### Evaluation Metrics

| Metric | Better when | Notes |
|---|---|---|
| Silhouette Score | Higher ↑ | Range −1 to 1; > 0.25 = weak, > 0.5 = strong |
| Davies-Bouldin Index (DBI) | Lower ↓ | 0 = perfect separation |
| Calinski-Harabasz Index (CH) | Higher ↑ | Ratio of between- to within-cluster scatter |

> **Note:** For DBSCAN, all metrics are computed on non-noise points only (`labels != -1`). This inflates silhouette when noise ratio is high — always check noise % alongside the score.

---

## Results Summary

### Original Feature Space (6-D)

| Algorithm | k / ε | Silhouette | DBI | CH | Noise |
|---|---|---|---|---|---|
| K-Means | k=2 | **0.2903** | 1.3515 | **189.05** | 0.0% |
| Agglomerative | k=2 | 0.2585 | 1.6004 | 134.62 | 0.0% |
| DBSCAN † | ε=0.7 | 0.3640 | 1.2523 | 73.25 | **77.95%** |

### PCA Space (2-D, 71.3% variance)

| Algorithm | k / ε | Silhouette | DBI | CH | Noise |
|---|---|---|---|---|---|
| K-Means | k=2 | **0.4082 ★** | **0.9724 ★** | **320.89 ★** | 0.0% |
| Agglomerative | k=2 | 0.4043 | 0.9962 | 292.35 | 0.0% |
| DBSCAN | ε=0.7 | 0.3387 | 0.6324 | 16.10 | 6.59% |

★ Best overall. † DBSCAN original-space silhouette computed on 22% of points — not directly comparable.

**Recommended model:** K-Means (k=2) on PCA-transformed data.

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
```

Install with:
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

The notebook was developed on **Google Colab** and loads data directly from Google Drive. No local data files are required.

---

## How to Run

1. Open `Untitled10-Copy1.ipynb` in Google Colab or Jupyter.
2. Run all cells top to bottom. The notebook is self-contained:
   - Data loads from the public Google Drive URL
   - All preprocessing, tuning, and evaluation runs sequentially
   - Cluster plots and selection plots are generated inline
3. To reproduce specific sections:

| Section | What it does |
|---|---|
| Cells 1–2 | Load data, EDA, normality tests, correlation heatmap |
| Cell 3 | Full hyperparameter tuning for all three algorithms |
| Cell 4 | Simplified comparative output table |
| Cell 5 | PCA projection + cluster scatter plots |

---

## Key Findings

- **Two clusters** is the optimal solution across all algorithms and both feature spaces — consistent with the known HoReCa (hospitality) vs. Retail channel split in the data.
- **PCA improves clustering quality** for all methods, not just visualisation. K-Means silhouette improves by 40.6% and DBI improves by 28% in PCA space.
- **DBSCAN is unsuitable for primary segmentation** on this dataset's original space (78% noise), but becomes practical after PCA reduction (6.6% noise). Its best use here is outlier/anomaly detection.
- **Metric disagreement matters:** DBSCAN's inflated silhouette (0.364) on original data is misleading — always pair silhouette with noise coverage when evaluating density-based methods.
- **Preprocessing is critical:** log1p + StandardScaler is not optional here; raw feature skewness of 4–8 makes Euclidean distances meaningless without it.
