"""
Day 3-4 — Train the classifier (the "semantic decoder")

Trains a small dense neural network on a landmarks CSV (63 columns +
label). Works identically on asl_landmarks.csv and include_landmarks.csv —
run it once per dataset, so you end up with two separate models:

    python train_classifier.py --input datasets/asl_landmarks.csv --output model_asl.keras --labels labels_asl.json
    python train_classifier.py --input datasets/include_landmarks.csv --output model_isl.keras --labels labels_isl.json

Outputs:
    - a .keras model file
    - a .json file mapping class index -> label string (needed at inference
      time, since LabelEncoder itself isn't saved)
    - printed accuracy, per-class precision/recall, and a confusion count
      for the worst-performing classes (so you have real numbers, not
      guesses, for the results table)

IMPORTANT — a real bug this script guards against:
    pandas will silently misread the 'label' column if some labels look
    numeric (e.g. the ASL letters '0'-'9'). Some rows get parsed as int,
    others as str, and "8" (int) and "8" (str) become two different
    classes — you'd never notice except accuracy looking mysteriously
    capped and one extra class appearing. This script forces
    dtype={'label': str} on read to prevent that. If you write any other
    script that touches these CSVs, do the same.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras


def load_dataset(csv_path):
    # dtype={'label': str} is load-bearing — see module docstring.
    df = pd.read_csv(csv_path, dtype={"label": str})
    feature_cols = [c for c in df.columns if c != "label"]
    if len(feature_cols) != 63:
        raise ValueError(
            f"Expected 63 landmark columns, found {len(feature_cols)}. "
            "Is this the right CSV / schema?"
        )
    X = df[feature_cols].values.astype("float32")
    y_raw = df["label"].values
    return X, y_raw, feature_cols


def build_model(input_dim, num_classes):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to landmarks CSV")
    parser.add_argument("--output", required=True, help="path to save model, e.g. model_asl.keras")
    parser.add_argument("--labels", required=True, help="path to save label mapping json, e.g. labels_asl.json")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input CSV not found: {args.input}")

    X, y_raw, feature_cols = load_dataset(args.input)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)

    print(f"Loaded {len(X)} rows, {num_classes} classes, {X.shape[1]} features.")

    # Some classes (esp. INCLUDE, ~14-21 samples/class) are small. Stratified
    # split still works as long as every class has >=2 samples; guard against
    # the rare dataset where that's not true, since stratify will crash otherwise.
    counts = pd.Series(y_raw).value_counts()
    too_small = counts[counts < 2]
    if len(too_small) > 0:
        raise SystemExit(
            f"These classes have fewer than 2 samples and can't be stratified: "
            f"{list(too_small.index)}. Collect more data for them or drop them."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    model = build_model(input_dim=X.shape[1], num_classes=num_classes)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=6, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop],
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n=== Final test accuracy: {test_acc*100:.2f}% (loss {test_loss:.4f}) ===\n")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0
    )

    # Print the 5 worst-performing classes by F1 so you know where the
    # model is actually weak, rather than just reporting one blended number.
    per_class_f1 = {
        cls: report[cls]["f1-score"] for cls in le.classes_ if cls in report
    }
    worst = sorted(per_class_f1.items(), key=lambda kv: kv[1])[:5]
    print("Weakest 5 classes (label: f1-score, support):")
    for cls, f1 in worst:
        support = int(report[cls]["support"])
        print(f"  {cls}: f1={f1:.2f}  n_test={support}")

    model.save(args.output)
    with open(args.labels, "w") as f:
        json.dump({int(i): cls for i, cls in enumerate(le.classes_)}, f, indent=2)

    print(f"\nSaved model to {args.output}")
    print(f"Saved label mapping to {args.labels}")


if __name__ == "__main__":
    main()