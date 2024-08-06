import random

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from minisom import MiniSom

from minisom2onnx import to_onnx

dim = 10
dlen = 1000
data = np.random.rand(dlen, 4)
target = [random.randint(1, 2) for i in range(dlen)]


@pytest.fixture
def setup_model():
    # Create a mock MiniSom model for testing
    model = MiniSom(dim, dim, data.shape[1])
    model.train_random(data, 1000)
    return model


def run_onnx_inference(onnx_model, input_data):
    # Create a runtime session
    session = ort.InferenceSession(onnx_model.SerializeToString())

    # Perform inference
    inputs = {session.get_inputs()[0].name: input_data}
    outputs = session.run(None, inputs)[0]

    return outputs


def test_onnx_check_model(setup_model):
    model = setup_model
    onnx_model = to_onnx(model, outputs=["winner"])
    try:
        onnx.checker.check_model(onnx_model)
    except:
        pytest.fail("Unexpected MyError ..")


def test_onnx_runtime_output(setup_model):
    model = setup_model
    onnx_model = to_onnx(model, outputs=["winner"])

    expected_output = np.array([model.winner(d) for d in data])
    onnx_output = run_onnx_inference(onnx_model, data)

    assert np.array_equal(expected_output, onnx_output)


def test_onnx_runtime_output_quantization(setup_model):
    model = setup_model
    onnx_model = to_onnx(model, outputs=["quantization"])

    expected_output = model.quantization(data)
    onnx_output = run_onnx_inference(onnx_model, data)

    assert np.allclose(expected_output, onnx_output)


def test_onnx_runtime_output_quantization_error(setup_model):
    model = setup_model
    onnx_model = to_onnx(model, outputs=["quantization_error"])

    expected_output = np.array([[model.quantization_error([d])] for d in data])
    onnx_output = run_onnx_inference(onnx_model, data)

    assert np.allclose(expected_output, onnx_output)


def test_onnx_runtime_output_distance(setup_model):
    model = setup_model
    onnx_model = to_onnx(model, outputs=["distance"])

    expected_output = np.array(
        [model._activation_distance(d, model._weights) for d in data]
    ).reshape(-1, dim * dim)
    onnx_output = run_onnx_inference(onnx_model, data)

    assert np.allclose(expected_output, onnx_output)


def test_onnx_runtime_output_label(setup_model):
    model = setup_model

    default_label = 0
    labels = np.full((dim, dim), fill_value=default_label, dtype=int)
    for position, counter in model.labels_map(data, target).items():
        labels[position] = max(counter, key=counter.get)

    onnx_model = to_onnx(model, labels=labels, outputs=["class"])

    winner = np.array([model.winner(d) for d in data])
    expected_output = np.array([labels[row, col] for row, col in winner])
    onnx_output = run_onnx_inference(onnx_model, data)

    assert np.array_equal(expected_output, onnx_output)


def test_onnx_runtime_output_outlier(setup_model):
    model = setup_model

    quantization_errors = np.array([model.quantization_error([x]) for x in data])
    threshold = np.percentile(quantization_errors, 95)

    onnx_model = to_onnx(model, threshold=threshold, outputs=["outlier"])

    quantization_error = np.array([[model.quantization_error([d])] for d in data])
    expected_output = quantization_error > threshold
    onnx_output = run_onnx_inference(onnx_model, data)

    assert np.array_equal(onnx_output.astype(bool), expected_output)
