"""Streamlit dashboard: predicting when a machine will fail (NASA C-MAPSS FD001).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import cmapss

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REQUIRED_FILES = [
    DATA_DIR / "test_FD001.txt",
    DATA_DIR / "RUL_FD001.txt",
    MODEL_DIR / "lstm.keras",
    MODEL_DIR / "scaler.joblib",
    MODEL_DIR / "test_predictions.csv",
]

st.set_page_config(page_title="Machine Failure Prediction", layout="wide")


@st.cache_resource(show_spinner="Loading LSTM model")
def load_model():
    import tensorflow as tf

    return tf.keras.models.load_model(MODEL_DIR / "lstm.keras")


@st.cache_resource
def load_scaler():
    return joblib.load(MODEL_DIR / "scaler.joblib")


@st.cache_data
def load_test_frame() -> pd.DataFrame:
    return cmapss.load_test(DATA_DIR)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    frame = pd.read_csv(MODEL_DIR / "test_predictions.csv")
    frame["status"] = frame["lstm_pred"].apply(cmapss.health_status)
    return frame


@st.cache_data
def load_json(name: str) -> dict | None:
    path = MODEL_DIR / name
    if not path.is_file():
        return None
    return json.loads(path.read_text())


@st.cache_data(show_spinner="Running inference")
def predict_trajectory(unit: int) -> pd.DataFrame:
    """Predict RUL at every cycle of one engine, as if monitoring it live."""
    frame = load_test_frame()
    unit_frame = frame[frame["unit"] == unit]
    windows = cmapss.unit_trajectory_windows(cmapss.scale(unit_frame, load_scaler()))
    predictions = np.clip(load_model().predict(windows, verbose=0).ravel(), 0, cmapss.RUL_CAP)
    return pd.DataFrame({
        "cycle": unit_frame["cycle"].to_numpy(),
        "Predicted RUL": predictions,
        "True RUL": unit_frame["rul"].to_numpy(),
    })


def ensure_artifacts() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_FILES if not path.exists()]
    if missing:
        st.error(
            "Required files are missing: " + ", ".join(missing)
            + ". Place the dataset in data/ and run `python train.py` before starting the dashboard."
        )
        st.stop()


def render_fleet(predictions: pd.DataFrame) -> None:
    counts = predictions["status"].value_counts()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Engines monitored", len(predictions))
    col2.metric("Critical", int(counts.get("Critical", 0)))
    col3.metric("Service soon", int(counts.get("Service Soon", 0)))
    col4.metric("Healthy", int(counts.get("Healthy", 0)))

    st.caption(
        f"Critical: predicted RUL below {cmapss.CRITICAL_THRESHOLD} cycles. "
        f"Service soon: below {cmapss.WARNING_THRESHOLD} cycles."
    )
    st.subheader("Predicted remaining life per engine")
    st.bar_chart(predictions.set_index("unit")["lstm_pred"], y_label="Predicted RUL (cycles)")

    st.subheader("Maintenance priority list")
    table = predictions.sort_values("lstm_pred")[["unit", "lstm_pred", "true_rul", "status"]]
    table = table.rename(columns={
        "unit": "Engine", "lstm_pred": "Predicted RUL", "true_rul": "Actual RUL", "status": "Status",
    })
    st.dataframe(table.round(1), hide_index=True, width="stretch")


def render_engine(predictions: pd.DataFrame) -> None:
    frame = load_test_frame()
    default_unit = int(predictions.sort_values("lstm_pred")["unit"].iloc[0])
    units = sorted(frame["unit"].unique().tolist())
    unit = st.selectbox("Select engine", units, index=units.index(default_unit))

    trajectory = predict_trajectory(int(unit))
    latest = trajectory.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted RUL (cycles)", f"{latest['Predicted RUL']:.0f}")
    col2.metric("Actual RUL (cycles)", f"{latest['True RUL']:.0f}")
    col3.metric("Status", cmapss.health_status(float(latest["Predicted RUL"])))

    st.subheader("Remaining life over the engine's history")
    st.line_chart(trajectory.set_index("cycle")[["Predicted RUL", "True RUL"]])
    st.caption(f"Actual RUL is capped at {cmapss.RUL_CAP} cycles, matching the training target.")

    st.subheader("Raw sensor readings")
    sensors = st.multiselect("Sensors", cmapss.SENSORS, default=["s4", "s11", "s12"])
    if sensors:
        unit_frame = frame[frame["unit"] == unit].set_index("cycle")
        st.line_chart(unit_frame[sensors])


def render_performance(predictions: pd.DataFrame) -> None:
    metrics = load_json("metrics.json")
    if metrics:
        st.subheader("Model comparison on the test set")
        table = pd.DataFrame(metrics["models"]).T[["rmse", "mae", "nasa_score", "train_seconds"]]
        table.columns = ["RMSE", "MAE", "NASA Score", "Train Time (s)"]
        st.dataframe(table.round(2), width="stretch")
        st.caption("Lower is better for all error metrics. The NASA score penalises late predictions more heavily.")

    st.subheader("Predicted vs actual RUL")
    scatter = predictions.rename(columns={"true_rul": "Actual RUL", "lstm_pred": "LSTM", "svr_pred": "SVR"})
    scatter = scatter.melt(id_vars="Actual RUL", value_vars=["LSTM", "SVR"], var_name="Model", value_name="Predicted RUL")
    st.scatter_chart(scatter, x="Actual RUL", y="Predicted RUL", color="Model")

    history = load_json("history.json")
    if history:
        st.subheader("LSTM training curve")
        curve = pd.DataFrame({"Training loss": history["loss"], "Validation loss": history["val_loss"]})
        curve.index.name = "Epoch"
        st.line_chart(curve)

    benchmark = load_json("benchmark.json")
    if benchmark:
        st.subheader("Edge deployment benchmark")
        table = pd.DataFrame(benchmark).T[["size_kb", "latency_ms", "rmse"]]
        table.columns = ["Size (KB)", "Latency (ms / sample)", "Test RMSE"]
        st.dataframe(table.round(3), width="stretch")
    else:
        st.info("Run `python export_tflite.py` to add the edge deployment benchmark.")


def main() -> None:
    ensure_artifacts()
    st.title("Machine Failure Prediction")
    st.write(
        "Predicts the remaining useful life (RUL) of turbofan engines from 14 sensor streams, "
        "using an LSTM network compared against an SVR baseline."
    )
    predictions = load_predictions()
    fleet_tab, engine_tab, performance_tab = st.tabs(["Fleet Overview", "Engine Detail", "Model Performance"])
    with fleet_tab:
        render_fleet(predictions)
    with engine_tab:
        render_engine(predictions)
    with performance_tab:
        render_performance(predictions)


main()
