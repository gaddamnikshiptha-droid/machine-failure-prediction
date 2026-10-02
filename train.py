"""Train an SVR baseline and an LSTM network to predict machine failure (RUL).

Usage:
    python train.py --data-dir data --model-dir models

Outputs (written to --model-dir):
    scaler.joblib          Sensor scaler fitted on training engines
    svr.joblib             SVR baseline pipeline
    lstm.keras             Trained LSTM model
    metrics.json           Test-set metrics for both models
    history.json           LSTM training curves
    test_predictions.csv   Per-engine predictions on the test set
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import cmapss

LOGGER = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--svr-samples", type=int, default=5000,
                        help="Maximum training windows for the SVR (SVR scales quadratically).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)


def train_svr(windows: np.ndarray, targets: np.ndarray, max_samples: int, seed: int):
    """Fit an RBF-kernel SVR on summary features extracted from each window."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    features = cmapss.summary_features(windows)
    if len(features) > max_samples:
        index = np.random.default_rng(seed).choice(len(features), max_samples, replace=False)
        features, targets = features[index], targets[index]

    model = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100.0, epsilon=2.0, gamma="scale"))
    model.fit(features, targets)
    return model


def build_lstm(window: int, n_features: int):
    """Two stacked LSTM layers followed by a small dense head.

    `unroll=True` turns the recurrence into a static graph, which makes the model
    convert cleanly to TensorFlow Lite for edge deployment. The final Rescaling
    layer lets the network learn in a [0, 1] range while outputting RUL in cycles.
    """
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(window, n_features), name="sensor_window")
    x = tf.keras.layers.LSTM(64, return_sequences=True, unroll=True)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(32, unroll=True)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dense(1)(x)
    outputs = tf.keras.layers.Rescaling(float(cmapss.RUL_CAP), name="rul")(x)

    model = tf.keras.Model(inputs, outputs, name="rul_lstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def train_lstm(args, x_train, y_train, x_val, y_val):
    import tensorflow as tf

    model = build_lstm(x_train.shape[1], x_train.shape[2])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )
    return model, history.history


def markdown_table(results: dict[str, dict]) -> str:
    lines = ["| Model | RMSE | MAE | NASA Score | Train Time (s) |", "|---|---|---|---|---|"]
    for name, values in results.items():
        lines.append(
            f"| {name} | {values['rmse']:.2f} | {values['mae']:.2f} | "
            f"{values['nasa_score']:.1f} | {values['train_seconds']:.1f} |"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading %s from %s", args.subset, args.data_dir)
    train_full = cmapss.load_train(args.data_dir, args.subset)
    test = cmapss.load_test(args.data_dir, args.subset)
    train_df, val_df = cmapss.split_units(train_full, args.val_fraction, args.seed)

    scaler = cmapss.fit_scaler(train_df)
    x_train, y_train = cmapss.make_windows(cmapss.scale(train_df, scaler))
    x_val, y_val = cmapss.make_windows(cmapss.scale(val_df, scaler))
    x_test, test_units, y_test = cmapss.last_windows(cmapss.scale(test, scaler))
    LOGGER.info("Windows - train: %s, val: %s, test: %s", x_train.shape, x_val.shape, x_test.shape)

    LOGGER.info("Training SVR baseline")
    start = time.perf_counter()
    svr = train_svr(x_train, y_train, args.svr_samples, args.seed)
    svr_seconds = time.perf_counter() - start
    svr_pred = np.clip(svr.predict(cmapss.summary_features(x_test)), 0, cmapss.RUL_CAP)

    LOGGER.info("Training LSTM")
    start = time.perf_counter()
    lstm, history = train_lstm(args, x_train, y_train, x_val, y_val)
    lstm_seconds = time.perf_counter() - start
    lstm_pred = np.clip(lstm.predict(x_test, verbose=0).ravel(), 0, cmapss.RUL_CAP)

    results = {
        "SVR": {**cmapss.regression_metrics(y_test, svr_pred), "train_seconds": svr_seconds},
        "LSTM": {
            **cmapss.regression_metrics(y_test, lstm_pred),
            "train_seconds": lstm_seconds,
            "epochs_trained": len(history["loss"]),
            "parameters": int(lstm.count_params()),
        },
    }
    metrics = {
        "subset": args.subset,
        "window": cmapss.WINDOW,
        "rul_cap": cmapss.RUL_CAP,
        "train_windows": int(len(x_train)),
        "test_engines": int(len(x_test)),
        "models": results,
    }

    joblib.dump(scaler, args.model_dir / "scaler.joblib")
    joblib.dump(svr, args.model_dir / "svr.joblib")
    lstm.save(args.model_dir / "lstm.keras")
    (args.model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.model_dir / "history.json").write_text(
        json.dumps({key: [float(v) for v in values] for key, values in history.items()}, indent=2)
    )
    pd.DataFrame({
        "unit": test_units,
        "true_rul": y_test,
        "svr_pred": svr_pred,
        "lstm_pred": lstm_pred,
    }).to_csv(args.model_dir / "test_predictions.csv", index=False)

    LOGGER.info("Artifacts saved to %s", args.model_dir.resolve())
    print("\n" + markdown_table(results) + "\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError) as error:
        LOGGER.error("%s", error)
        sys.exit(1)
