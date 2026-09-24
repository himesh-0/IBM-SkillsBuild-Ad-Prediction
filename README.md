# Social Network Ads — Binary Classification ML App

A **single-file** Python web application that trains five binary-classification models on the Social Network Ads dataset and exposes an interactive browser UI for real-time purchase predictions.

---

## 📁 Project Structure

```
social_network_ads_ml/
├── app.py               ← All backend ML + Flask frontend (one file)
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
└── Social_Network_Ads.csv  ← Place your dataset here (or upload via UI)
```

---

## 🧠 Dataset

| Column | Description |
|--------|-------------|
| `User ID` | Unique user identifier (dropped before training) |
| `Gender` | Male / Female (label-encoded) |
| `Age` | User age (feature) |
| `EstimatedSalary` | Annual salary estimate in USD (feature) |
| `Purchased` | **Target** — 1 = purchased, 0 = did not purchase |

- **400 records** total
- **Binary classification** task
- Features used for training: **Age** and **EstimatedSalary**

---

## 🤖 Models Trained

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline linear classifier |
| K-Nearest Neighbors | Distance-based (k=5) |
| Support Vector Machine | RBF kernel with probability estimates |
| Random Forest | 100 estimators ensemble |
| Gradient Boosting | 100 estimators boosted ensemble |

All models are scaled with `StandardScaler` before training.

---

## 🖥️ Web UI Pages

| Route | Description |
|-------|-------------|
| `/` | Home — upload CSV and trigger training |
| `/dashboard` | Metrics table, charts (ROC, CM, boundary, distributions) |
| `/predict` | Single-user and bulk purchase prediction |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

### 3. Train the models

1. Go to the **Home** page.
2. Click **Choose File** and upload `Social_Network_Ads.csv`.
3. Click **⚡ Train All Models**.
4. You will be redirected to the **Dashboard** automatically.

### 4. Predict

Navigate to **Predict**, enter an Age and Estimated Salary, choose a classifier, and click **🔮 Predict**.

---

## 📊 Evaluation Metrics

- **Test Accuracy** — percentage of correct predictions on the 25 % hold-out set
- **AUC-ROC** — area under the Receiver Operating Characteristic curve
- **CV Accuracy** — mean 5-fold cross-validation accuracy

---

## 📈 Charts Generated

| Chart | Description |
|-------|-------------|
| Model Comparison | Side-by-side bar chart of Accuracy, AUC, and CV score |
| ROC Curves | All five models on one plot |
| Confusion Matrix | For the best-performing model |
| Decision Boundary | 2-D visualisation of the decision region |
| Feature Distributions | Age and Salary histograms split by class |

---

## ⚙️ Configuration

All configuration lives at the top of `app.py`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `FEATURE_COLS` | `["Age", "EstimatedSalary"]` | Input features |
| `TARGET_COL` | `"Purchased"` | Target column |
| Test split | 25 % | `train_test_split(test_size=0.25)` |
| CV folds | 5 | `cross_val_score(cv=5)` |

---

## 🛠️ Requirements

| Package | Version |
|---------|---------|
| Flask | ≥ 3.0 |
| NumPy | ≥ 1.26 |
| pandas | ≥ 2.2 |
| scikit-learn | ≥ 1.4 |
| matplotlib | ≥ 3.8 |
| seaborn | ≥ 0.13 |

---

## 📝 License

MIT — free to use, modify, and distribute.
