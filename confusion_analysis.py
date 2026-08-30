"""
Confusion matrix / per-letter error analysis for the ASL classifier.

Uses the SAME train/test split logic as train_classifier.py (same
random_state=42) so results are reproducible and match your reported
accuracy. Produces:

    - confusion_matrix.png       full 36x36 heatmap (all classes)
    - a printed list of the top confused pairs, e.g. "1 predicted as T: 4 times"

This is what should back up any "we found X is confused with Y" claim in
your submission's Testing/Implementation Status section — real numbers
from a real held-out test set, not anecdotes from live testing.

Run:
    python confusion_analysis.py --input datasets/asl_landmarks.csv
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras


def load_dataset(csv_path):
    df = pd.read_csv(csv_path, dtype={"label": str})  # same dtype fix as train_classifier.py
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype("float32")
    y_raw = df["label"].values
    return X, y_raw


def build_model(input_dim, num_classes):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_image", default="confusion_matrix.png")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--top_confusions", type=int, default=10)
    args = parser.parse_args()

    X, y_raw = load_dataset(args.input)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42  # same split as train_classifier.py
    )

    model = build_model(X.shape[1], num_classes)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=args.epochs, batch_size=32, callbacks=[early_stop], verbose=2)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    test_acc = (y_pred == y_test).mean()
    print(f"\n=== Test accuracy: {test_acc*100:.2f}% ===\n")

    cm = confusion_matrix(y_test, y_pred)

    # Save a heatmap image for the paper.
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(le.classes_, fontsize=7, rotation=90)
    ax.set_yticklabels(le.classes_, fontsize=7)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"ASL Classifier Confusion Matrix (test accuracy {test_acc*100:.1f}%)")
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(args.output_image, dpi=150)
    print(f"Saved confusion matrix heatmap to {args.output_image}")

    # Find the biggest OFF-DIAGONAL confusions (i.e. actual mistakes, not correct predictions).
    confusions = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], le.classes_[i], le.classes_[j]))
    confusions.sort(reverse=True)

    print(f"\n=== Top {args.top_confusions} confused pairs (true -> predicted, count) ===")
    for count, true_label, pred_label in confusions[:args.top_confusions]:
        print(f"  {true_label} misread as {pred_label}: {count} times")

    if not confusions:
        print("  No confusions in the test set — model got every class exactly right.")


if __name__ == "__main__":
    main()