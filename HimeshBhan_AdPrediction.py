"""
Social Network Ads — Binary Classification
==========================================
Single-file application: ML backend + Flask web UI.

Dataset columns expected in Social_Network_Ads.csv:
  User ID, Gender, Age, EstimatedSalary, Purchased

Run:
    python app.py
Then open: http://127.0.0.1:5000
"""

import os
import io
import base64
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend — must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from flask import Flask, request, render_template_string, redirect, url_for

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0.  Global state (loaded once at startup)
# ---------------------------------------------------------------------------
MODELS = {}          # name -> fitted model
SCALER = None
FEATURE_COLS = ["Age", "EstimatedSalary"]
TARGET_COL   = "Purchased"
REPORT_DATA  = {}    # holds metrics / chart PNGs (base64)

# ---------------------------------------------------------------------------
# 1.  Helper: figure → base64 PNG
# ---------------------------------------------------------------------------
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ---------------------------------------------------------------------------
# 2.  Training pipeline
# ---------------------------------------------------------------------------
def train_pipeline(csv_path: str):
    global MODELS, SCALER, REPORT_DATA

    # --- Load & preprocess ---
    df = pd.read_csv(csv_path)

    # Drop User ID if present
    if "User ID" in df.columns:
        df.drop(columns=["User ID"], inplace=True)

    # Encode Gender
    if "Gender" in df.columns:
        le = LabelEncoder()
        df["Gender"] = le.fit_transform(df["Gender"])

    df.dropna(inplace=True)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    SCALER = scaler

    # --- Define classifiers ---
    classifiers = {
        "Logistic Regression":      LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors":      KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine":   SVC(probability=True, random_state=42),
        "Random Forest":            RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":        GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    metrics = []

    for name, clf in classifiers.items():
        clf.fit(X_train_sc, y_train)
        MODELS[name] = clf

        y_pred  = clf.predict(X_test_sc)
        y_proba = clf.predict_proba(X_test_sc)[:, 1]

        acc   = accuracy_score(y_test, y_pred)
        auc   = roc_auc_score(y_test, y_proba)
        cv    = cross_val_score(clf, scaler.transform(X), y, cv=5, scoring="accuracy").mean()

        metrics.append({
            "Model":    name,
            "Accuracy": round(acc * 100, 2),
            "AUC-ROC":  round(auc, 4),
            "CV Acc":   round(cv * 100, 2),
            "clf":      clf,
            "y_pred":   y_pred,
            "y_proba":  y_proba,
        })

    metrics_df = pd.DataFrame(metrics).drop(columns=["clf", "y_pred", "y_proba"])
    best_row   = max(metrics, key=lambda r: r["AUC-ROC"])
    best_name  = best_row["Model"]
    best_clf   = best_row["clf"]

    # ---------- Charts ----------

    # 1. Model comparison bar chart
    fig, ax = plt.subplots(figsize=(9, 4))
    x  = np.arange(len(metrics))
    w  = 0.28
    ax.bar(x - w, [m["Accuracy"] for m in metrics], w, label="Test Accuracy (%)", color="#3b82d4")
    ax.bar(x,     [m["AUC-ROC"]*100 for m in metrics], w, label="AUC-ROC × 100", color="#7c5cd8")
    ax.bar(x + w, [m["CV Acc"] for m in metrics], w, label="CV Accuracy (%)", color="#10b981")
    ax.set_xticks(x)
    ax.set_xticklabels([m["Model"] for m in metrics], rotation=15, ha="right", fontsize=9)
    ax.set_ylim(50, 105)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    chart_comparison = fig_to_b64(fig)
    plt.close(fig)

    # 2. Confusion matrix for best model
    cm = confusion_matrix(y_test, best_row["y_pred"])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Purchased", "Purchased"],
                yticklabels=["Not Purchased", "Purchased"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name}")
    fig.tight_layout()
    chart_cm = fig_to_b64(fig)
    plt.close(fig)

    # 3. ROC curves
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ["#3b82d4", "#7c5cd8", "#ef4444", "#10b981", "#f59e0b"]
    for i, m in enumerate(metrics):
        fpr, tpr, _ = roc_curve(y_test, m["y_proba"])
        ax.plot(fpr, tpr, lw=1.8, color=colors[i],
                label=f"{m['Model']} (AUC={m['AUC-ROC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    chart_roc = fig_to_b64(fig)
    plt.close(fig)

    # 4. Decision boundary for best model (Age vs Salary)
    h = 0.02
    x_min, x_max = X_test_sc[:, 0].min() - 1, X_test_sc[:, 0].max() + 1
    y_min, y_max = X_test_sc[:, 1].min() - 1, X_test_sc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = best_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
    scatter = ax.scatter(X_test_sc[:, 0], X_test_sc[:, 1],
                         c=y_test, cmap=plt.cm.RdBu, edgecolors="k",
                         linewidths=0.4, s=30, alpha=0.9)
    patch0 = mpatches.Patch(color=plt.cm.RdBu(0.1),  label="Not Purchased")
    patch1 = mpatches.Patch(color=plt.cm.RdBu(0.9),  label="Purchased")
    ax.legend(handles=[patch0, patch1], fontsize=8)
    ax.set_xlabel("Age (scaled)")
    ax.set_ylabel("Estimated Salary (scaled)")
    ax.set_title(f"Decision Boundary — {best_name}")
    fig.tight_layout()
    chart_boundary = fig_to_b64(fig)
    plt.close(fig)

    # 5. Feature distribution
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for col, ax in zip(FEATURE_COLS, axes):
        purchased     = df[df[TARGET_COL] == 1][col]
        not_purchased = df[df[TARGET_COL] == 0][col]
        ax.hist(not_purchased, bins=25, alpha=0.6, color="#3b82d4", label="Not Purchased")
        ax.hist(purchased,     bins=25, alpha=0.6, color="#ef4444",  label="Purchased")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.set_title(f"{col} Distribution")
        ax.legend(fontsize=8)
    fig.tight_layout()
    chart_dist = fig_to_b64(fig)
    plt.close(fig)

    # ---------- Classification report for best model ----------
    report_text = classification_report(
        y_test, best_row["y_pred"],
        target_names=["Not Purchased", "Purchased"]
    )

    REPORT_DATA = {
        "metrics_df":        metrics_df,
        "metrics_list":      metrics,
        "best_name":         best_name,
        "best_acc":          best_row["Accuracy"],
        "best_auc":          best_row["AUC-ROC"],
        "report_text":       report_text,
        "chart_comparison":  chart_comparison,
        "chart_cm":          chart_cm,
        "chart_roc":         chart_roc,
        "chart_boundary":    chart_boundary,
        "chart_dist":        chart_dist,
        "dataset_rows":      len(df),
        "train_rows":        len(X_train),
        "test_rows":         len(X_test),
    }
    return True


# ---------------------------------------------------------------------------
# 3.  HTML templates (inline)
# ---------------------------------------------------------------------------
BASE_STYLE = """
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
         font-size: 14px; background: #f0f2f5; color: #1f2328; }
  a { color: #3b82d4; text-decoration: none; }
  a:hover { text-decoration: underline; }

  /* nav */
  nav { background: #1e293b; color: #fff; padding: 0 24px;
        display: flex; align-items: center; gap: 24px; height: 52px; }
  nav .brand { font-size: 16px; font-weight: 700; color: #fff; }
  nav a { color: #cbd5e1; font-size: 13px; }
  nav a:hover { color: #fff; text-decoration: none; }

  /* container */
  .container { max-width: 1080px; margin: 0 auto; padding: 28px 20px; }

  /* card */
  .card { background: #fff; border: 1px solid #e5e7eb;
          border-radius: 10px; padding: 24px; margin-bottom: 24px; }
  .card h2 { margin: 0 0 16px; font-size: 17px; color: #1e293b; }

  /* table */
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { background: #f7f8fa; text-align: left; padding: 8px 12px;
       border-bottom: 2px solid #e5e7eb; color: #57606a; font-weight: 600; }
  td { padding: 8px 12px; border-bottom: 1px solid #f1f3f5; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #f7f8fa; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
           font-size: 11px; font-weight: 600; }
  .badge-best { background: #dcfce7; color: #15803d; }

  /* form */
  .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }
  label { display: block; font-size: 12px; font-weight: 600;
          color: #57606a; margin-bottom: 4px; }
  input, select { width: 100%; padding: 8px 10px; border: 1px solid #d1d5db;
                  border-radius: 6px; font-size: 14px; }
  input:focus, select:focus { outline: none; border-color: #3b82d4;
                               box-shadow: 0 0 0 3px rgba(59,130,212,0.15); }
  .btn { display: inline-block; padding: 9px 22px; border: none;
         border-radius: 6px; font-size: 14px; font-weight: 600;
         cursor: pointer; transition: opacity 0.15s; }
  .btn-primary { background: #3b82d4; color: #fff; }
  .btn-primary:hover { opacity: 0.88; }
  .btn-secondary { background: #7c5cd8; color: #fff; }
  .btn-secondary:hover { opacity: 0.88; }

  /* result badge */
  .result-box { padding: 20px 24px; border-radius: 8px; margin-top: 20px;
                display: flex; align-items: center; gap: 16px; }
  .result-yes { background: #dcfce7; border: 1px solid #86efac; }
  .result-no  { background: #fee2e2; border: 1px solid #fca5a5; }
  .result-icon { font-size: 32px; }
  .result-label { font-size: 18px; font-weight: 700; }
  .result-sub   { font-size: 13px; color: #57606a; margin-top: 2px; }

  /* chart grid */
  .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .chart-grid img { width: 100%; border-radius: 6px; border: 1px solid #e5e7eb; }
  @media (max-width: 640px) { .chart-grid { grid-template-columns: 1fr; } }

  /* stat pills */
  .stats { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
  .stat { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px;
          padding: 16px 24px; flex: 1; min-width: 140px; }
  .stat .val { font-size: 26px; font-weight: 700; color: #3b82d4; }
  .stat .lbl { font-size: 12px; color: #57606a; margin-top: 2px; }

  /* upload zone */
  .upload-zone { border: 2px dashed #d1d5db; border-radius: 10px;
                 padding: 40px; text-align: center; background: #fafafa; }
  .upload-zone p { margin: 8px 0; color: #57606a; }
  pre { background: #f7f8fa; border: 1px solid #e5e7eb; border-radius: 6px;
        padding: 14px; font-size: 12px; overflow-x: auto; line-height: 1.5; }
  .alert { padding: 12px 16px; border-radius: 6px; margin-bottom: 16px; font-size: 13px; }
  .alert-success { background: #dcfce7; border: 1px solid #86efac; color: #15803d; }
  .alert-error   { background: #fee2e2; border: 1px solid #fca5a5; color: #991b1b; }
</style>
"""

TEMPLATE_INDEX = """
<!DOCTYPE html><html lang="en"><head>
{{ style|safe }}
<title>Social Network Ads — ML Classifier</title>
</head><body>
<nav>
  <span class="brand">🧠 Social Network Ads ML</span>
  <a href="/">Home</a>
  <a href="/dashboard">Dashboard</a>
  <a href="/predict">Predict</a>
</nav>
<div class="container">
  {% if msg %}
  <div class="alert {{ 'alert-success' if ok else 'alert-error' }}">{{ msg }}</div>
  {% endif %}

  <div class="card">
    <h2>Welcome</h2>
    <p>This application trains multiple binary-classification models on the
       <strong>Social Network Ads</strong> dataset and lets you predict whether a
       social-media user will purchase a product based on their
       <em>Age</em> and <em>Estimated Salary</em>.</p>
    <p>Upload your <code>Social_Network_Ads.csv</code> file below to train the models,
       then head to the <a href="/dashboard">Dashboard</a> to view metrics, or
       <a href="/predict">Predict</a> for a single-user inference.</p>
  </div>

  <div class="card">
    <h2>Train Models</h2>
    <form method="POST" action="/train" enctype="multipart/form-data">
      <div class="upload-zone">
        <p style="font-size:15px;font-weight:600;">Upload CSV File</p>
        <p>Expected columns: <code>User ID, Gender, Age, EstimatedSalary, Purchased</code></p>
        <br>
        <input type="file" name="csvfile" accept=".csv" required style="width:auto;">
        <br><br>
        <button class="btn btn-primary" type="submit">⚡ Train All Models</button>
      </div>
    </form>
  </div>

  {% if trained %}
  <div class="alert alert-success">
    ✅ Models are trained and ready. Go to
    <a href="/dashboard">Dashboard</a> or <a href="/predict">Predict</a>.
  </div>
  {% endif %}
</div>
</body></html>
"""

TEMPLATE_DASHBOARD = """
<!DOCTYPE html><html lang="en"><head>
{{ style|safe }}
<title>Dashboard — Social Network Ads ML</title>
</head><body>
<nav>
  <span class="brand">🧠 Social Network Ads ML</span>
  <a href="/">Home</a>
  <a href="/dashboard">Dashboard</a>
  <a href="/predict">Predict</a>
</nav>
<div class="container">
  <div class="stats">
    <div class="stat"><div class="val">{{ rd.dataset_rows }}</div><div class="lbl">Total Samples</div></div>
    <div class="stat"><div class="val">{{ rd.train_rows }}</div><div class="lbl">Training Samples</div></div>
    <div class="stat"><div class="val">{{ rd.test_rows }}</div><div class="lbl">Test Samples</div></div>
    <div class="stat"><div class="val">{{ rd.best_acc }}%</div><div class="lbl">Best Accuracy ({{ rd.best_name }})</div></div>
    <div class="stat"><div class="val">{{ rd.best_auc }}</div><div class="lbl">Best AUC-ROC</div></div>
  </div>

  <div class="card">
    <h2>Model Performance</h2>
    <table>
      <thead><tr><th>Model</th><th>Test Accuracy (%)</th><th>AUC-ROC</th><th>CV Accuracy (%)</th></tr></thead>
      <tbody>
        {% for row in rd.metrics_list %}
        <tr>
          <td>{{ row.Model }}
            {% if row.Model == rd.best_name %}
            <span class="badge badge-best">Best</span>
            {% endif %}
          </td>
          <td>{{ row.Accuracy }}</td>
          <td>{{ row['AUC-ROC'] }}</td>
          <td>{{ row['CV Acc'] }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Model Comparison Chart</h2>
    <img src="data:image/png;base64,{{ rd.chart_comparison }}" alt="Model Comparison">
  </div>

  <div class="card">
    <h2>Charts</h2>
    <div class="chart-grid">
      <div>
        <p style="font-weight:600;margin-bottom:8px;">ROC Curves</p>
        <img src="data:image/png;base64,{{ rd.chart_roc }}" alt="ROC Curves">
      </div>
      <div>
        <p style="font-weight:600;margin-bottom:8px;">Confusion Matrix (Best Model)</p>
        <img src="data:image/png;base64,{{ rd.chart_cm }}" alt="Confusion Matrix">
      </div>
      <div>
        <p style="font-weight:600;margin-bottom:8px;">Feature Distributions</p>
        <img src="data:image/png;base64,{{ rd.chart_dist }}" alt="Feature Distribution">
      </div>
      <div>
        <p style="font-weight:600;margin-bottom:8px;">Decision Boundary (Best Model)</p>
        <img src="data:image/png;base64,{{ rd.chart_boundary }}" alt="Decision Boundary">
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Classification Report — {{ rd.best_name }}</h2>
    <pre>{{ rd.report_text }}</pre>
  </div>
</div>
</body></html>
"""

TEMPLATE_PREDICT = """
<!DOCTYPE html><html lang="en"><head>
{{ style|safe }}
<title>Predict — Social Network Ads ML</title>
</head><body>
<nav>
  <span class="brand">🧠 Social Network Ads ML</span>
  <a href="/">Home</a>
  <a href="/dashboard">Dashboard</a>
  <a href="/predict">Predict</a>
</nav>
<div class="container">
  <div class="card">
    <h2>Single-User Purchase Prediction</h2>
    <p>Enter the user details below and choose a model to predict whether the user will purchase a product.</p>
    <form method="POST" action="/predict">
      <div class="form-grid">
        <div>
          <label for="age">Age</label>
          <input type="number" id="age" name="age" min="18" max="70"
                 value="{{ age or 30 }}" required>
        </div>
        <div>
          <label for="salary">Estimated Salary ($)</label>
          <input type="number" id="salary" name="salary" min="10000" max="200000"
                 step="1000" value="{{ salary or 50000 }}" required>
        </div>
        <div>
          <label for="model_name">Classifier</label>
          <select id="model_name" name="model_name">
            {% for name in model_names %}
            <option value="{{ name }}" {{ 'selected' if name == chosen_model }}>{{ name }}</option>
            {% endfor %}
          </select>
        </div>
      </div>
      <br>
      <button class="btn btn-primary" type="submit">🔮 Predict</button>
      {% if result is not none %}
      <div class="result-box {{ 'result-yes' if result == 1 else 'result-no' }}">
        <div class="result-icon">{{ '🛒' if result == 1 else '🚫' }}</div>
        <div>
          <div class="result-label">
            {{ 'Likely to Purchase' if result == 1 else 'Unlikely to Purchase' }}
          </div>
          <div class="result-sub">
            Model: <strong>{{ chosen_model }}</strong> &nbsp;|&nbsp;
            Age: <strong>{{ age }}</strong> &nbsp;|&nbsp;
            Salary: <strong>${{ "{:,}".format(salary|int) }}</strong>
            &nbsp;|&nbsp; Probability: <strong>{{ proba }}%</strong>
          </div>
        </div>
      </div>
      {% endif %}
    </form>
  </div>

  <div class="card">
    <h2>Bulk Predict (Paste CSV rows)</h2>
    <p style="color:#57606a;font-size:13px;">Format: one row per line — <code>Age,EstimatedSalary</code></p>
    <form method="POST" action="/bulk_predict">
      <textarea name="bulk_csv" rows="6"
        style="width:100%;padding:8px;border:1px solid #d1d5db;border-radius:6px;font-size:13px;font-family:monospace;"
        placeholder="30,50000&#10;45,80000&#10;22,35000">{{ bulk_csv or '' }}</textarea>
      <br><br>
      <div style="display:flex;gap:12px;align-items:center;">
        <select name="bulk_model" style="width:240px;padding:8px 10px;border:1px solid #d1d5db;border-radius:6px;">
          {% for name in model_names %}
          <option value="{{ name }}" {{ 'selected' if name == chosen_model }}>{{ name }}</option>
          {% endfor %}
        </select>
        <button class="btn btn-secondary" type="submit">📋 Bulk Predict</button>
      </div>
    </form>
    {% if bulk_results %}
    <br>
    <table>
      <thead><tr><th>#</th><th>Age</th><th>Salary ($)</th><th>Prediction</th><th>Probability</th></tr></thead>
      <tbody>
        {% for r in bulk_results %}
        <tr>
          <td>{{ loop.index }}</td>
          <td>{{ r.age }}</td>
          <td>{{ "{:,}".format(r.salary) }}</td>
          <td>
            <span class="badge {{ 'badge-best' if r.pred == 1 else '' }}"
                  style="{{ 'background:#fee2e2;color:#991b1b;' if r.pred == 0 else '' }}">
              {{ '✅ Purchase' if r.pred == 1 else '❌ No Purchase' }}
            </span>
          </td>
          <td>{{ r.proba }}%</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% endif %}
  </div>
</div>
</body></html>
"""

TEMPLATE_NOT_TRAINED = """
<!DOCTYPE html><html lang="en"><head>
{{ style|safe }}
<title>Not Trained</title>
</head><body>
<nav>
  <span class="brand">🧠 Social Network Ads ML</span>
  <a href="/">Home</a>
</nav>
<div class="container">
  <div class="card">
    <h2>Models not trained yet</h2>
    <p>Please <a href="/">upload the CSV and train the models</a> first.</p>
  </div>
</div>
</body></html>
"""


# ---------------------------------------------------------------------------
# 4.  Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "social-ads-ml-secret"


@app.route("/")
def index():
    return render_template_string(
        TEMPLATE_INDEX,
        style=BASE_STYLE,
        trained=bool(MODELS),
        msg=None, ok=True,
    )


@app.route("/train", methods=["POST"])
def train():
    csv_file = request.files.get("csvfile")
    if not csv_file:
        return render_template_string(
            TEMPLATE_INDEX, style=BASE_STYLE, trained=False,
            msg="No file uploaded.", ok=False,
        )
    tmp_path = os.path.join(os.path.dirname(__file__), "_tmp_upload.csv")
    csv_file.save(tmp_path)
    try:
        train_pipeline(tmp_path)
        os.remove(tmp_path)
        return redirect(url_for("dashboard"))
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return render_template_string(
            TEMPLATE_INDEX, style=BASE_STYLE, trained=False,
            msg=f"Training failed: {exc}", ok=False,
        )


@app.route("/dashboard")
def dashboard():
    if not MODELS:
        return render_template_string(TEMPLATE_NOT_TRAINED, style=BASE_STYLE)
    return render_template_string(
        TEMPLATE_DASHBOARD,
        style=BASE_STYLE,
        rd=REPORT_DATA,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not MODELS:
        return render_template_string(TEMPLATE_NOT_TRAINED, style=BASE_STYLE)

    result = None
    proba  = None
    age    = None
    salary = None
    chosen_model = REPORT_DATA.get("best_name", list(MODELS.keys())[0])

    if request.method == "POST":
        try:
            age    = int(request.form["age"])
            salary = int(request.form["salary"])
            chosen_model = request.form.get("model_name", chosen_model)
            clf = MODELS[chosen_model]
            X_input = SCALER.transform([[age, salary]])
            result  = int(clf.predict(X_input)[0])
            proba   = round(clf.predict_proba(X_input)[0][result] * 100, 1)
        except Exception as exc:
            result = None

    return render_template_string(
        TEMPLATE_PREDICT,
        style=BASE_STYLE,
        model_names=list(MODELS.keys()),
        result=result,
        proba=proba,
        age=age,
        salary=salary,
        chosen_model=chosen_model,
        bulk_results=None,
        bulk_csv="",
    )


@app.route("/bulk_predict", methods=["POST"])
def bulk_predict():
    if not MODELS:
        return render_template_string(TEMPLATE_NOT_TRAINED, style=BASE_STYLE)

    raw      = request.form.get("bulk_csv", "")
    bulk_model = request.form.get("bulk_model", REPORT_DATA.get("best_name"))
    clf      = MODELS.get(bulk_model, list(MODELS.values())[0])
    results  = []

    for line in raw.strip().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        try:
            age_v    = int(parts[0])
            salary_v = int(parts[1])
            X_in     = SCALER.transform([[age_v, salary_v]])
            pred     = int(clf.predict(X_in)[0])
            prob     = round(clf.predict_proba(X_in)[0][pred] * 100, 1)
            results.append({"age": age_v, "salary": salary_v, "pred": pred, "proba": prob})
        except Exception:
            continue

    return render_template_string(
        TEMPLATE_PREDICT,
        style=BASE_STYLE,
        model_names=list(MODELS.keys()),
        result=None, proba=None, age=None, salary=None,
        chosen_model=bulk_model,
        bulk_results=results,
        bulk_csv=raw,
    )


# ---------------------------------------------------------------------------
# 5.  Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Social Network Ads — Binary Classification App")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)
