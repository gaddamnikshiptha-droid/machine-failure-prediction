"""Convert the trained LSTM to TensorFlow Lite and benchmark it for edge deployment.

Produces two variants:
    lstm_float32.tflite   Direct conversion, no quantization
    lstm_int8.tflite      Dynamic-range quantization (INT8 weights)

Each variant is compared with the original Keras model on file size, single-sample
latency and test-set RMSE. Results are written to models/benchmark.json and a
combined summary is written to RESULTS.md.

Usage:
    python export_tflite.py --data-dir data --model-dir models
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

import cmapss

LOGGER = logging.getLogger("export_tflite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--runs", type=int, default=200, help="Timed inference runs per model.")
    return parser.parse_args()


def export_saved_model(model: tf.keras.Model, export_dir: Path) -> None:
    """Export an inference-only SavedModel, compatible with Keras 2 and Keras 3."""
    if hasattr(model, "export"):
        model.export(str(export_dir))
    else:
        tf.saved_model.save(model, str(export_dir))


def convert(saved_model_dir: Path, quantize: bool) -> tuple[bytes, bool]:
    """Convert to TFLite, falling back to TF Select ops if builtins are insufficient."""

    def build(select_ops: bool) -> bytes:
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if select_ops:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter._experimental_lower_tensor_list_ops = False
        return converter.convert()

    try:
        return build(select_ops=False), False
    except Exception as error:  # The converter raises several unrelated exception types.
        LOGGER.warning("Builtin-only conversion failed (%s). Retrying with TF Select ops.", error)
        return build(select_ops=True), True


def time_calls(call, runs: int) -> float:
    """Mean latency in milliseconds after a short warm-up."""
    for _ in range(10):
        call()
    start = time.perf_counter()
    for _ in range(runs):
        call()
    return (time.perf_counter() - start) / runs * 1000.0


def benchmark_keras(model: tf.keras.Model, x_test: np.ndarray, runs: int) -> tuple[np.ndarray, float]:
    signature = tf.TensorSpec(shape=(1, *x_test.shape[1:]), dtype=tf.float32)
    infer = tf.function(lambda x: model(x, training=False), input_signature=[signature])
    predictions = np.array([float(infer(sample[None, ...]).numpy().ravel()[0]) for sample in x_test])
    sample = tf.constant(x_test[:1])
    latency = time_calls(lambda: infer(sample), runs)
    return predictions, latency


def benchmark_tflite(model_bytes: bytes, x_test: np.ndarray, runs: int) -> tuple[np.ndarray, float]:
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.resize_tensor_input(input_index, [1, *x_test.shape[1:]])
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_index = interpreter.get_output_details()[0]["index"]

    def run(sample: np.ndarray) -> float:
        interpreter.set_tensor(input_details["index"], sample[None, ...].astype(input_details["dtype"]))
        interpreter.invoke()
        return float(interpreter.get_tensor(output_index).ravel()[0])

    predictions = np.array([run(sample) for sample in x_test])
    latency = time_calls(lambda: run(x_test[0]), runs)
    return predictions, latency


def write_results_markdown(path: Path, metrics: dict | None, benchmark: dict) -> None:
    lines = ["# Results", "", "Dataset: NASA C-MAPSS FD001 (100 test engines). RUL capped at 125 cycles.", ""]
    if metrics:
        lines += [
            "## Model Comparison",
            "",
            "| Model | RMSE | MAE | NASA Score | Train Time (s) |",
            "|---|---|---|---|---|",
        ]
        for name, values in metrics["models"].items():
            lines.append(
                f"| {name} | {values['rmse']:.2f} | {values['mae']:.2f} | "
                f"{values['nasa_score']:.1f} | {values['train_seconds']:.1f} |"
            )
        lines.append("")
    lines += [
        "## Edge Deployment Benchmark",
        "",
        "| Variant | Size (KB) | Latency (ms / sample) | Test RMSE |",
        "|---|---|---|---|",
    ]
    for name, values in benchmark.items():
        lines.append(
            f"| {name} | {values['size_kb']:.1f} | {values['latency_ms']:.3f} | {values['rmse']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    keras_path = args.model_dir / "lstm.keras"
    scaler_path = args.model_dir / "scaler.joblib"
    for required in (keras_path, scaler_path):
        if not required.is_file():
            raise FileNotFoundError(f"{required} not found. Run train.py first.")

    model = tf.keras.models.load_model(keras_path)
    scaler = joblib.load(scaler_path)
    test = cmapss.load_test(args.data_dir, args.subset)
    x_test, _, y_test = cmapss.last_windows(cmapss.scale(test, scaler))

    benchmark: dict[str, dict] = {}

    LOGGER.info("Benchmarking Keras model")
    predictions, latency = benchmark_keras(model, x_test, args.runs)
    benchmark["Keras (float32)"] = {
        "size_kb": keras_path.stat().st_size / 1024.0,
        "latency_ms": latency,
        "rmse": cmapss.regression_metrics(y_test, np.clip(predictions, 0, cmapss.RUL_CAP))["rmse"],
    }

    with tempfile.TemporaryDirectory() as tmp:
        saved_model_dir = Path(tmp) / "saved_model"
        export_saved_model(model, saved_model_dir)

        for label, filename, quantize in (
            ("TFLite (float32)", "lstm_float32.tflite", False),
            ("TFLite (INT8 weights)", "lstm_int8.tflite", True),
        ):
            LOGGER.info("Converting %s", label)
            model_bytes, used_select_ops = convert(saved_model_dir, quantize)
            (args.model_dir / filename).write_bytes(model_bytes)
            predictions, latency = benchmark_tflite(model_bytes, x_test, args.runs)
            benchmark[label] = {
                "size_kb": len(model_bytes) / 1024.0,
                "latency_ms": latency,
                "rmse": cmapss.regression_metrics(y_test, np.clip(predictions, 0, cmapss.RUL_CAP))["rmse"],
                "select_tf_ops": used_select_ops,
            }

    (args.model_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2))

    metrics_path = args.model_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else None
    write_results_markdown(Path("RESULTS.md"), metrics, benchmark)

    for name, values in benchmark.items():
        LOGGER.info(
            "%-22s size=%8.1f KB  latency=%7.3f ms  rmse=%6.2f",
            name, values["size_kb"], values["latency_ms"], values["rmse"],
        )
    LOGGER.info("Summary written to RESULTS.md")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError) as error:
        LOGGER.error("%s", error)
        sys.exit(1)
