from pathlib import Path
from datetime import datetime
import json
import sqlite3
import warnings
import shap
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash

BASE = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE / "data" / "pune_crop_yield_part_1.csv",
    BASE / "data" / "pune_crop_yield_part_1(2).csv",
    BASE / "pune_crop_yield_part_1.csv",
    BASE.parent / "data" / "pune_crop_yield_part_1.csv",
    BASE.parent / "pune_crop_yield_part_1.csv",
]
DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), DATA_CANDIDATES[0])
MODEL_DIR = BASE / "models"
DB_PATH = BASE / "cropyield_history.db"
MODEL_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = "replace-this-local-development-secret"

CROPS = ["Onion", "Sugarcane"]
TARGET = "yield_tonnes_per_ha"
EXCLUDE = [TARGET, "record_type", "production_tonnes", "estimated_revenue_inr"]
RANDOM_STATE = 42
MODEL_PATHS = {
    "Onion": MODEL_DIR / "onion_yield_model.joblib",
    "Sugarcane": MODEL_DIR / "sugarcane_yield_model.joblib",
}
models = {}
metrics = {}
dataset_status = {"loaded": False, "rows": 0, "": None, "message": "Dataset not loaded"}
dataset = None


def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                crop TEXT NOT NULL,
                predicted_yield REAL NOT NULL,
                inputs_json TEXT NOT NULL,
                explanation_json TEXT
            )
        """)


def load_dataset():
    global dataset, dataset_status
    # Also search the project and its immediate parent for common filename variants.
    candidates = list(DATA_CANDIDATES)
    for folder in (BASE / "data", BASE, BASE.parent / "data", BASE.parent):
        if folder.exists():
            candidates.extend(folder.glob("pune_crop_yield_part_1*.csv"))
    source_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if source_path is None:
        dataset_status = {
            "loaded": False, "rows": 0, "": None,
            "message": "Dataset not found. Place pune_crop_yield_part_1.csv inside the project's data folder."
        }
        return None
    try:
        frame = pd.read_csv(source_path)
        if "crop" not in frame.columns or TARGET not in frame.columns:
            raise ValueError("Dataset must contain 'crop' and 'yield_tonnes_per_ha' columns.")
        frame["crop"] = frame["crop"].astype(str).str.strip().str.title()
        frame = frame[frame["crop"].isin(CROPS)].copy()
        if frame.empty:
            raise ValueError("No Onion or Sugarcane rows found. Check the crop column values.")
        is_ = False
        if "record_type" in frame.columns:
            is_ = frame["record_type"].astype(str).str.contains("", case=False).any()
        dataset = frame
        dataset_status = {
            "loaded": True, "rows": int(len(frame)), "": bool(is_),
            "message": f"Loaded {source_path.name}"
        }
        return frame
    except Exception as exc:
        dataset_status = {
            "loaded": False, "rows": 0, "": None,
            "message": f"Dataset load failed: {exc}"
        }
        return None


def train_models(frame):
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    trained = {}
    scorecard = {}
    for crop in CROPS:
        part = frame[frame["crop"] == crop].dropna(subset=[TARGET]).copy()
        if len(part) < 10:
            continue
        X = part.drop(columns=EXCLUDE, errors="ignore")
        y = pd.to_numeric(part[TARGET], errors="coerce")
        valid = y.notna()
        X, y = X.loc[valid], y.loc[valid]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE
        )
        numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()
        numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        preprocessor = ColumnTransformer([
            ("num", numeric, numeric_cols),
            ("cat", categorical, categorical_cols)
        ])
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", ExtraTreesRegressor(
                n_estimators=100, min_samples_leaf=2,
                random_state=RANDOM_STATE, n_jobs=-1
            ))
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        trained[crop] = pipeline
        scorecard[crop] = {
            "records": int(len(part)),
            "mae": round(float(mean_absolute_error(y_test, predictions)), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, predictions))), 3),
            "r2": round(float(r2_score(y_test, predictions)), 4)
        }
        joblib.dump(pipeline, MODEL_PATHS[crop])
    return trained, scorecard


def load_or_train():
    global models, metrics
    # Always load the dataset for dashboard/analytics even when saved models exist.
    frame = load_dataset()
    if all(path.exists() for path in MODEL_PATHS.values()):
        try:
            models = {crop: joblib.load(path) for crop, path in MODEL_PATHS.items()}
            metric_path = MODEL_DIR / "metrics.json"
            if metric_path.exists():
                metrics = json.loads(metric_path.read_text(encoding="utf-8"))
            if models:
                return
        except Exception as exc:
            warnings.warn(f"Saved models could not be loaded; will retrain if data is available. {exc}")
    if frame is not None:
        models, metrics = train_models(frame)
        (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def feature_metadata(crop):
    if crop not in models:
        return []
    pipeline = models[crop]
    pre = pipeline.named_steps["preprocess"]
    original = list(getattr(pre, "feature_names_in_", []))
    if dataset is None:
        return [{"name": n, "type": "text"} for n in original]
    source = dataset[dataset["crop"] == crop]
    result = []
    for name in original:
        if name in source.columns and pd.api.types.is_numeric_dtype(source[name]):
            vals = pd.to_numeric(source[name], errors="coerce").dropna()
            result.append({
                "name": name, "type": "number",
                "placeholder": f"e.g. {round(float(vals.median()), 2)}" if len(vals) else "Enter number"
            })
        else:
            choices = []
            if name in source.columns:
                choices = source[name].dropna().astype(str).value_counts().head(30).index.tolist()
            result.append({"name": name, "type": "select" if choices and len(choices) <= 15 else "text",
                           "choices": choices if choices and len(choices) <= 15 else [],
                           "placeholder": "Choose value" if choices and len(choices) <= 15 else "Enter value"})
    return result


def build_input_frame(crop, submitted):
    expected = list(getattr(models[crop].named_steps["preprocess"], "feature_names_in_", []))
    source = dataset[dataset["crop"] == crop] if dataset is not None else None
    row = {}
    for name in expected:
        raw = submitted.get(name, "")
        if raw is None or str(raw).strip() == "":
            row[name] = np.nan
            continue
        is_numeric = source is not None and name in source.columns and pd.api.types.is_numeric_dtype(source[name])
        if is_numeric:
            try:
                row[name] = float(raw)
            except (ValueError, TypeError):
                raise ValueError(f"'{name}' must be a number.")
        else:
            row[name] = str(raw).strip()
    return pd.DataFrame([row], columns=expected)


def calculate_shap(crop, X):
    # Import SHAP only when the user requests an explanation.
    pipeline = models[crop]

    preprocessor = pipeline.named_steps["preprocess"]
    tree_model = pipeline.named_steps["model"]

    # Transform the input using the trained preprocessing pipeline.
    transformed = preprocessor.transform(X)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed)

    # Get feature names after preprocessing.
    names = [
        str(name).replace("num__", "").replace("cat__", "")
        for name in preprocessor.get_feature_names_out()
    ]

    # Calculate SHAP values.
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(transformed)

    values = np.asarray(shap_values).reshape(-1)

    # Get the model's baseline prediction.
    base = explainer.expected_value

    if isinstance(base, (list, np.ndarray)):
        base = np.asarray(base).reshape(-1)[0]

    # Prepare the most influential features.
    top_features = [
        {
            "feature": names[i],
            "impact": round(float(values[i]), 5)
        }
        for i in range(min(len(names), len(values)))
    ]

    top_features = sorted(
        top_features,
        key=lambda item: abs(item["impact"]),
        reverse=True
    )[:12]

    return {
        "base_value": float(base),
        "top_features": top_features,
        "note": (
            "Positive SHAP values push the prediction above "
            "the model baseline; negative values push it below."
        )
    }

def history_rows(limit=100):
    with connect_db() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    result = []
    for row in rows:
        result.append({
            "id": row["id"], "created_at": row["created_at"],
            "crop": row["crop"], "predicted_yield": row["predicted_yield"],
            "inputs": json.loads(row["inputs_json"]),
            "explanation": json.loads(row["explanation_json"]) if row["explanation_json"] else None
        })
    return result


@app.route("/")
def dashboard():
    history = history_rows(6)
    all_history = history_rows(500)
    by_crop = {crop: sum(1 for item in all_history if item["crop"] == crop) for crop in CROPS}
    latest_yield = all_history[0]["predicted_yield"] if all_history else None
    return render_template(
        "dashboard.html", active="dashboard", crops=CROPS, models=models,
        metrics=metrics, status=dataset_status, history=history,
        total_predictions=len(all_history), by_crop=by_crop, latest_yield=latest_yield
    )


@app.route("/predict")
def predict_page():
    return render_template("predict.html", active="predict", crops=CROPS, models=models)


@app.route("/history")
def history_page():
    return render_template("history.html", active="history", history=history_rows(200))


@app.route("/analytics")
def analytics_page():
    frame = dataset if dataset is not None else pd.DataFrame()
    crop_counts = frame["crop"].value_counts().reindex(CROPS, fill_value=0).to_dict() if not frame.empty else {c: 0 for c in CROPS}
    yield_summary = {}
    if not frame.empty:
        for crop in CROPS:
            vals = pd.to_numeric(frame.loc[frame["crop"] == crop, TARGET], errors="coerce").dropna()
            yield_summary[crop] = {
                "mean": round(float(vals.mean()), 2) if len(vals) else None,
                "median": round(float(vals.median()), 2) if len(vals) else None,
                "min": round(float(vals.min()), 2) if len(vals) else None,
                "max": round(float(vals.max()), 2) if len(vals) else None,
                "count": int(len(vals))
            }
    return render_template("analytics.html", active="analytics", crops=CROPS,
                           crop_counts=crop_counts, yield_summary=yield_summary,
                           metrics=metrics, status=dataset_status)


@app.route("/about")
def about_page():
    return render_template("about.html", active="about", status=dataset_status)


@app.route("/api/status")
def api_status():
    return jsonify({
        "status": "ok",
        "available_models": list(models.keys()),
        "dataset": dataset_status,
        "metrics": metrics
    })


@app.route("/api/features/<crop>")
def api_features(crop):
    if crop not in CROPS:
        return jsonify({"error": "Choose Onion or Sugarcane."}), 400
    if crop not in models:
        return jsonify({"error": f"{crop} model is unavailable. Check the dataset and training setup."}), 503
    return jsonify({"crop": crop, "features": feature_metadata(crop)})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    crop = payload.get("crop", "")
    submitted = payload.get("features", {})
    if crop not in CROPS:
        return jsonify({"error": "Select Onion or Sugarcane."}), 400
    if crop not in models:
        return jsonify({"error": f"The {crop} model is unavailable. Check the dataset and models folder."}), 503
    try:
        X = build_input_frame(crop, submitted)
        prediction = float(models[crop].predict(X)[0])
        created = datetime.now().isoformat(timespec="seconds")
        with connect_db() as conn:
            cur = conn.execute(
                "INSERT INTO predictions(created_at,crop,predicted_yield,inputs_json) VALUES(?,?,?,?)",
                (created, crop, prediction, json.dumps(submitted))
            )
            prediction_id = cur.lastrowid
        return jsonify({
            "id": prediction_id, "crop": crop,
            "predicted_yield_tonnes_per_ha": round(prediction, 3),
            "unit": "tonnes/ha", "created_at": created
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/explain", methods=["POST"])
def api_explain():
    payload = request.get_json(silent=True) or {}
    crop = payload.get("crop", "")
    submitted = payload.get("features", {})
    prediction_id = payload.get("prediction_id")
    if crop not in models:
        return jsonify({"error": f"No trained model found for {crop}."}), 503
    try:
        X = build_input_frame(crop, submitted)
        explanation = calculate_shap(crop, X)
        if prediction_id:
            with connect_db() as conn:
                conn.execute(
                    "UPDATE predictions SET explanation_json=? WHERE id=?",
                    (json.dumps(explanation), prediction_id)
                )
        return jsonify({"crop": crop, **explanation})
    except Exception as exc:
        return jsonify({"error": f"SHAP explanation failed: {exc}"}), 400


@app.route("/api/history")
def api_history():
    return jsonify(history_rows(200))


@app.route("/api/history/<int:prediction_id>", methods=["DELETE"])
def delete_history(prediction_id):
    with connect_db() as conn:
        conn.execute("DELETE FROM predictions WHERE id=?", (prediction_id,))
    return jsonify({"deleted": True})


@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    with connect_db() as conn:
        conn.execute("DELETE FROM predictions")
    return jsonify({"cleared": True})


init_db()
load_or_train()

if __name__ == "__main__":
    app.run(debug=True)
