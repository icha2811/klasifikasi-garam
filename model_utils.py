"""Logika penelitian Random Forest garam dari notebook 4 tahapan.

Menggunakan tiga stratified holdout tunggal (tanpa CV atau hyperparameter tuning).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

FEATURES = ["K_ug_g", "Na_mg_g", "Ca_ug_g", "Mg_ug_g", "Fe_ug_g", "Mn_ug_g", "Zn_ug_g", "Cu_ug_g"]
SPLITS = {"60:40": 0.40, "70:30": 0.30, "80:20": 0.20}
COUNTRY_NAMES = {"Brazil": "Brasil", "France": "Prancis", "Portugal": "Portugal", "Spain": "Spanyol"}
FEATURE_LABELS = {"K_ug_g": "Kalium (K), µg/g", "Na_mg_g": "Natrium (Na), mg/g", "Ca_ug_g": "Kalsium (Ca), µg/g", "Mg_ug_g": "Magnesium (Mg), µg/g", "Fe_ug_g": "Besi (Fe), µg/g", "Mn_ug_g": "Mangan (Mn), µg/g", "Zn_ug_g": "Seng (Zn), µg/g", "Cu_ug_g": "Tembaga (Cu), µg/g"}


def load_dataset(path: str | Path = "table_salt_origin.csv") -> pd.DataFrame:
    data = pd.read_csv(path)
    required = ["Sample", *FEATURES, "Country"]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {', '.join(sorted(missing))}")
    if data[required].isna().any().any():
        raise ValueError("Dataset memiliki nilai kosong. Periksa file CSV.")
    if not data["Sample"].is_unique:
        raise ValueError("Kolom Sample berisi ID duplikat.")
    for name in FEATURES:
        data[name] = pd.to_numeric(data[name], errors="raise")
    return data


def fit_three_scenarios(data: pd.DataFrame):
    x, y = data[FEATURES], data["Country"]
    labels = sorted(y.unique().tolist())
    results = {}
    summary_rows = []
    for name, ratio in SPLITS.items():
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=ratio, stratify=y, random_state=42
        )
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        cm = confusion_matrix(y_test, predicted, labels=labels)
        results[name] = {
            "model": model, "x_test": x_test, "y_test": y_test,
            "predicted": predicted, "cm": cm,
            "train_size": len(x_train), "test_size": len(x_test),
        }
        summary_rows.append({
            "Skenario": name, "Data latih": len(x_train), "Data uji": len(x_test),
            "Accuracy (%)": 100 * accuracy_score(y_test, predicted),
            "Precision macro (%)": 100 * precision_score(y_test, predicted, average="macro", zero_division=0),
            "Recall macro (%)": 100 * recall_score(y_test, predicted, average="macro", zero_division=0),
            "F1 macro (%)": 100 * f1_score(y_test, predicted, average="macro", zero_division=0),
        })
    summary = pd.DataFrame(summary_rows)
    return results, summary, labels


def classification_by_country(result, labels):
    rows = []
    for country in labels:
        actual = (result["y_test"] == country)
        predicted = (result["predicted"] == country)
        tp = int((actual & predicted).sum())
        fp = int((~actual & predicted).sum())
        fn = int((actual & ~predicted).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        rows.append({"Negara": COUNTRY_NAMES[country], "Precision (%)": 100 * precision,
                     "Recall (%)": 100 * recall, "F1-score (%)": 100 * f1,
                     "Support": int(actual.sum())})
    return pd.DataFrame(rows)


def predict_from_minerals(model, values: dict):
    frame = pd.DataFrame([{key: float(values[key]) for key in FEATURES}], columns=FEATURES)
    prediction = model.predict(frame)[0]
    probs = model.predict_proba(frame)[0]
    prob_table = pd.DataFrame({"Negara": [COUNTRY_NAMES[c] for c in model.classes_], "Proporsi suara/probabilitas model": probs})
    return COUNTRY_NAMES[prediction], prob_table.sort_values("Proporsi suara/probabilitas model", ascending=False)
