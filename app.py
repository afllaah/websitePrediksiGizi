from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os, io, json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-before-deploying")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

EXPECTED = ["Umur (bulan)", "Jenis Kelamin", "Tinggi Badan (cm)", "Status Gizi"]
FEATURES = ["Umur (bulan)", "Jenis Kelamin", "Tinggi Badan (cm)"]
MODEL = {"ready": False, "message": "Unggah data_balita.csv untuk melatih model."}
HISTORY = []

def normalize_gender(series):
    raw = series.astype("string").str.strip().str.lower()
    mapping = {
        "laki-laki": 0, "laki laki": 0, "laki_laki": 0, "l": 0, "male": 0, "m": 0,
        "perempuan": 1, "p": 1, "female": 1, "f": 1
    }
    return raw.map(mapping)

def load_and_train(path):
    global MODEL
    df = pd.read_csv(path)
    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        raise ValueError("Kolom wajib tidak ditemukan: " + ", ".join(missing) +
                         ". Kolom yang dibutuhkan: " + ", ".join(EXPECTED))
    df = df[EXPECTED].copy()
    df["Umur (bulan)"] = pd.to_numeric(df["Umur (bulan)"], errors="coerce")
    df["Tinggi Badan (cm)"] = pd.to_numeric(df["Tinggi Badan (cm)"], errors="coerce")
    df["Jenis Kelamin"] = normalize_gender(df["Jenis Kelamin"])
    df["Status Gizi"] = df["Status Gizi"].astype("string").str.strip().str.lower()
    df.loc[df["Umur (bulan)"] < 0, "Umur (bulan)"] = np.nan
    df.loc[df["Tinggi Badan (cm)"] <= 0, "Tinggi Badan (cm)"] = np.nan
    df.loc[~df["Status Gizi"].notna() | (df["Status Gizi"] == ""), "Status Gizi"] = pd.NA

    # Target kosong tidak dapat dipakai untuk supervised learning.
    train_df = df.dropna(subset=["Status Gizi"]).copy()
    if len(train_df) < 10:
        raise ValueError("Dataset perlu minimal 10 baris dengan label Status Gizi yang terisi.")
    if train_df["Status Gizi"].nunique() < 2:
        raise ValueError("Model klasifikasi memerlukan minimal 2 kategori Status Gizi.")
    if train_df["Jenis Kelamin"].isna().all():
        raise ValueError("Kolom Jenis Kelamin tidak terbaca. Gunakan kategori laki-laki/perempuan.")
    X = train_df[FEATURES].copy()
    y = train_df["Status Gizi"].astype(str)
    # Drop rows with missing gender; numeric missing values are imputed with training medians.
    valid_gender = X["Jenis Kelamin"].notna()
    X, y = X.loc[valid_gender], y.loc[valid_gender]
    if y.nunique() < 2 or len(y) < 10:
        raise ValueError("Data valid terlalu sedikit setelah pembersihan.")
    # Split can be stratified only when every class has at least 2 rows.
    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    medians = X_train[["Umur (bulan)", "Tinggi Badan (cm)"]].median()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[["Umur (bulan)", "Tinggi Badan (cm)"]] = X_train[["Umur (bulan)", "Tinggi Badan (cm)"]].fillna(medians)
    X_test[["Umur (bulan)", "Tinggi Badan (cm)"]] = X_test[["Umur (bulan)", "Tinggi Badan (cm)"]].fillna(medians)
    if X_train.isna().any().any():
        raise ValueError("Ada data fitur yang tidak dapat dibersihkan. Periksa kolom umur, jenis kelamin, dan tinggi.")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Pilih K secara aman. Cross-validation tidak dijalankan jika kelas terlalu kecil.
    max_k = max(1, min(15, len(X_train)))
    candidate_ks = list(range(1, max_k + 1))
    best_k = 1
    best_cv = None
    min_class_train = int(y_train.value_counts().min())
    if min_class_train >= 2 and len(y_train) >= 5:
        cv = min(5, min_class_train)
        for k in candidate_ks:
            try:
                scores = cross_val_score(
                    KNeighborsClassifier(n_neighbors=k), X_train_scaled, y_train,
                    cv=cv, scoring="accuracy"
                )
                score = float(scores.mean())
                if best_cv is None or score > best_cv:
                    best_cv, best_k = score, k
            except Exception:
                continue
    else:
        # If cross-validation is not feasible, use a conservative valid K.
        best_k = min(3, len(X_train))
    model = KNeighborsClassifier(n_neighbors=min(best_k, len(X_train)))
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, pred, labels=labels).tolist()
    report = classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0)
    clean_display = df.copy()
    clean_display["Jenis Kelamin"] = clean_display["Jenis Kelamin"].map({0: "laki-laki", 1: "perempuan"})
    clean_display = clean_display.replace({np.nan: None, pd.NA: None})
    MODEL = {
        "ready": True, "df": df, "display_df": clean_display, "scaler": scaler, "model": model,
        "medians": medians, "features": FEATURES, "labels": labels,
        "rows": int(len(df)), "usable_rows": int(len(train_df)), "classes": int(y.nunique()),
        "missing": int(df.isna().sum().sum()), "duplicates": int(df.duplicated().sum()),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().items()},
        "accuracy": float(accuracy_score(y_test, pred)),
        "train_accuracy": float(accuracy_score(y_train, model.predict(X_train_scaled))),
        "test_size": int(len(y_test)), "train_size": int(len(y_train)), "best_k": int(model.n_neighbors),
        "cv_accuracy": best_cv, "confusion": cm, "report": report,
        "source_name": os.path.basename(path), "message": "Model berhasil dilatih dari dataset yang diunggah."
    }
    return MODEL

def serializable_dataset(limit=500):
    if not MODEL.get("ready"):
        return []
    df = MODEL["display_df"].head(limit).copy()
    df = df.where(pd.notna(df), None)
    return df.to_dict(orient="records")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def status():
    if not MODEL.get("ready"):
        return jsonify({"ready": False, "message": MODEL.get("message")})
    return jsonify({
        "ready": True, "message": MODEL["message"], "source_name": MODEL["source_name"],
        "rows": MODEL["rows"], "usable_rows": MODEL["usable_rows"], "classes": MODEL["classes"],
        "missing": MODEL["missing"], "duplicates": MODEL["duplicates"],
        "accuracy": MODEL["accuracy"], "train_accuracy": MODEL["train_accuracy"],
        "test_size": MODEL["test_size"], "train_size": MODEL["train_size"],
        "best_k": MODEL["best_k"], "cv_accuracy": MODEL["cv_accuracy"],
        "class_counts": MODEL["class_counts"], "labels": MODEL["labels"],
        "confusion": MODEL["confusion"], "report": MODEL["report"]
    })

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Pilih file CSV terlebih dahulu."}), 400
    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Format file harus CSV."}), 400
    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)
    try:
        result = load_and_train(path)
        return jsonify({"ok": True, "message": result["message"], "source_name": result["source_name"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/predict", methods=["POST"])
def predict():
    if not MODEL.get("ready"):
        return jsonify({"error": "Model belum tersedia. Unggah dataset CSV yang valid terlebih dahulu."}), 400
    data = request.get_json(silent=True) or {}
    try:
        age = float(data.get("age"))
        gender = int(data.get("gender"))
        height = float(data.get("height"))
        if not np.isfinite(age) or not np.isfinite(height) or age < 0 or age > 60 or height <= 0 or height > 150:
            raise ValueError("Pastikan umur 0–60 bulan dan tinggi badan lebih dari 0 sampai 150 cm.")
        if gender not in (0, 1):
            raise ValueError("Jenis kelamin tidak valid.")
        row = pd.DataFrame([{"Umur (bulan)": age, "Jenis Kelamin": gender, "Tinggi Badan (cm)": height}])
        scaled = MODEL["scaler"].transform(row)
        result = str(MODEL["model"].predict(scaled)[0])
        history_item = {
            "waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "umur_bulan": age, "jenis_kelamin": "Laki-laki" if gender == 0 else "Perempuan",
            "tinggi_cm": height, "prediksi": result
        }
        HISTORY.insert(0, history_item)
        del HISTORY[200:]
        return jsonify({"prediction": result, "history": history_item,
                        "note": "Ini prediksi model berdasarkan dataset, bukan diagnosis medis."})
    except (TypeError, ValueError) as e:
        return jsonify({"error": str(e) or "Input tidak valid."}), 400

@app.route("/api/dataset")
def dataset():
    if not MODEL.get("ready"):
        return jsonify({"rows": [], "total": 0})
    return jsonify({"rows": serializable_dataset(), "total": MODEL["rows"]})

@app.route("/api/history")
def history():
    return jsonify({"rows": HISTORY})

@app.route("/api/history.csv")
def history_csv():
    df = pd.DataFrame(HISTORY)
    if df.empty:
        df = pd.DataFrame(columns=["waktu", "umur_bulan", "jenis_kelamin", "tinggi_cm", "prediksi"])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    mem = io.BytesIO(buf.getvalue().encode("utf-8-sig"))
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="riwayat_prediksi_balita.csv")

if __name__ == "__main__":
    # Bila data_balita.csv ada di folder project, model dilatih otomatis saat aplikasi mulai.
    default_csv = os.path.join(os.path.dirname(__file__), "data_balita.csv")
    if os.path.exists(default_csv):
        try:
            load_and_train(default_csv)
            print("Model berhasil dilatih dari data_balita.csv")
        except Exception as exc:
            print("Dataset awal belum dapat digunakan:", exc)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
