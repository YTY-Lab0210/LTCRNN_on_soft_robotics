import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


class LTCNeuron(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        init_w = keras.initializers.RandomNormal(stddev=0.5)

        self.w = self.add_weight(
            name="w", shape=(input_dim, self.units), initializer=init_w, trainable=True
        )
        self.r = self.add_weight(
            name="r", shape=(input_dim, self.units), initializer="ones", trainable=True
        )
        self.mu = self.add_weight(
            name="mu", shape=(input_dim, self.units), initializer="zeros", trainable=True
        )

    def call(self, inputs, states):
        x = states[0]
        delta_t = 0.01

        inputs_expanded = tf.expand_dims(inputs, axis=-1)
        sigma = tf.math.sigmoid(inputs_expanded * self.r + self.mu)
        damping = 1.0 + tf.reduce_sum(tf.abs(self.w) * sigma, axis=1)
        driving = tf.reduce_sum(self.w * sigma, axis=1)
        dx = -damping * x + driving
        x_new = x + delta_t * dx
        return x_new, [x_new]

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def format_vector(name, arr):
    values = ", ".join(f"{float(v): .8f}f" for v in arr)
    return f"const float {name}[{len(arr)}] = {{\n  {values}\n}};"


def format_matrix(name, arr):
    rows = []
    for row in arr:
        values = ", ".join(f"{float(v): .8f}f" for v in row)
        rows.append(f"  {{{values}}}")
    return f"const float {name}[{arr.shape[0]}][{arr.shape[1]}] = {{\n" + ",\n".join(rows) + "\n};"


def find_ltc_and_dense(model):
    ltc_weights = None
    dense_weights = None

    for layer in model.layers:
        if isinstance(layer, keras.layers.RNN) and isinstance(layer.cell, LTCNeuron):
            ltc_weights = layer.get_weights()
        if isinstance(layer, keras.layers.Dense):
            dense_weights = layer.get_weights()

    if ltc_weights is None:
        raise RuntimeError("Could not find keras.layers.RNN(LTCNeuron) in the model.")
    if dense_weights is None:
        raise RuntimeError("Could not find the final Dense layer in the model.")
    if len(ltc_weights) != 3:
        raise RuntimeError(f"Expected 3 LTC arrays [w, r, mu], got {len(ltc_weights)}.")
    if len(dense_weights) != 2:
        raise RuntimeError(f"Expected 2 Dense arrays [kernel, bias], got {len(dense_weights)}.")

    return ltc_weights, dense_weights


def main():
    parser = argparse.ArgumentParser(
        description="Export trained LTC-4 Keras weights as Arduino C arrays."
    )
    parser.add_argument("model_path", type=Path, help="Saved Keras model path (.keras or .h5).")
    parser.add_argument("--output", type=Path, help="Optional text file for exported C arrays.")
    args = parser.parse_args()

    model = keras.models.load_model(
        args.model_path,
        custom_objects={"LTCNeuron": LTCNeuron},
        compile=False,
    )

    (ltc_w, ltc_r, ltc_mu), (dense_w, dense_b) = find_ltc_and_dense(model)

    chunks = [
        "// Paste these arrays into ltc4_zscore_inference.ino",
        format_matrix("LTC_W", np.asarray(ltc_w)),
        "",
        format_matrix("LTC_R", np.asarray(ltc_r)),
        "",
        format_matrix("LTC_MU", np.asarray(ltc_mu)),
        "",
        format_matrix("DENSE_W", np.asarray(dense_w)),
        "",
        format_vector("DENSE_B", np.asarray(dense_b)),
    ]
    text = "\n".join(chunks)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")

    print(text)


if __name__ == "__main__":
    main()
