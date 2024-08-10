import random

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from minisom import MiniSom

from minisom2onnx import to_onnx

dim1 = 10
dlen1 = 1000
data1 = np.random.rand(dlen1, 4)
target1 = [random.randint(1, 2) for i in range(dlen1)]


@pytest.fixture
def setup_model():
    # Create a mock MiniSom model for testing
    model = MiniSom(dim1, dim1, data1.shape[1])
    model.train_random(data1, 1000)
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
    onnx.checker.check_model(onnx_model)


def test_onnx_runtime_output(setup_model):
    model = setup_model
    onnx_model = to_onnx(model, outputs=["winner"])

    expected_output = np.array([model.winner(d) for d in data1])
    onnx_output = run_onnx_inference(onnx_model, data1)

    assert np.array_equal(expected_output, onnx_output)


def test_onnx_runtime_output_weights(setup_model):
    model = setup_model
    onnx_model = to_onnx(model, outputs=["weights"])

    expected_output = model.get_weights()
    onnx_output = run_onnx_inference(onnx_model, data1)

    assert np.allclose(expected_output, onnx_output)


def test_onnx_runtime_output_quantization(setup_model):
    model = setup_model
    onnx_model = to_onnx(model, outputs=["quantization"])

    expected_output = model.quantization(data1)
    onnx_output = run_onnx_inference(onnx_model, data1)

    assert np.allclose(expected_output, onnx_output)


def test_onnx_runtime_output_quantization_error(setup_model):
    model = setup_model
    onnx_model = to_onnx(model, outputs=["quantization_error"])

    expected_output = np.array([[model.quantization_error([d])] for d in data1])
    onnx_output = run_onnx_inference(onnx_model, data1)

    assert np.allclose(expected_output, onnx_output)


def test_onnx_runtime_output_distance(setup_model):
    model = setup_model
    onnx_model = to_onnx(model, outputs=["distance"])

    expected_output = np.array(
        [model._activation_distance(d, model._weights) for d in data1]
    ).reshape(-1, dim1 * dim1)
    onnx_output = run_onnx_inference(onnx_model, data1)

    assert np.allclose(expected_output, onnx_output)


def test_onnx_runtime_output_label(setup_model):
    model = setup_model

    default_label = 0
    labels = np.full((dim1, dim1), fill_value=default_label, dtype=int)
    for position, counter in model.labels_map(data1, target1).items():
        labels[position] = max(counter, key=counter.get)

    onnx_model = to_onnx(model, labels=labels, outputs=["class"])

    winner = np.array([model.winner(d) for d in data1])
    expected_output = np.array([labels[row, col] for row, col in winner])
    onnx_output = run_onnx_inference(onnx_model, data1)

    assert np.array_equal(expected_output, onnx_output)


def test_onnx_runtime_output_outlier(setup_model):
    model = setup_model

    quantization_errors = np.array([model.quantization_error([x]) for x in data1])
    threshold = np.percentile(quantization_errors, 95)

    onnx_model = to_onnx(model, threshold=threshold, outputs=["outlier"])

    quantization_error = np.array([[model.quantization_error([d])] for d in data1])
    expected_output = quantization_error > threshold
    onnx_output = run_onnx_inference(onnx_model, data1)

    assert np.array_equal(onnx_output.astype(bool), expected_output)


#########################################################################################


dim2 = 30
dlen2 = 10000
data2 = np.random.rand(dlen2, 10)
target2 = [random.randint(1, 2) for i in range(dlen2)]


@pytest.fixture
def setup_model2():
    # Create a mock MiniSom model for testing
    model = MiniSom(
        dim2, dim2, data2.shape[1], topology="hexagonal", neighborhood_function="bubble"
    )
    model.train(data2, 1000, random_order=True)
    return model


def test_onnx_check_model2(setup_model2):
    model = setup_model2
    onnx_model = to_onnx(model, outputs=["winner"])
    onnx.checker.check_model(onnx_model)


def test_onnx_runtime_output2(setup_model2):
    model = setup_model2
    onnx_model = to_onnx(model, outputs=["winner"])

    expected_output = np.array([model.winner(d) for d in data2])
    onnx_output = run_onnx_inference(onnx_model, data2)

    assert np.array_equal(expected_output, onnx_output)


def test_onnx_runtime_output_weights2(setup_model2):
    model = setup_model2
    onnx_model = to_onnx(model, outputs=["weights"])

    expected_output = model.get_weights()
    onnx_output = run_onnx_inference(onnx_model, data2)

    assert np.allclose(expected_output, onnx_output)


def test_onnx_runtime_output_quantization2(setup_model2):
    model = setup_model2
    onnx_model = to_onnx(model, outputs=["quantization"])

    expected_output = model.quantization(data2)
    onnx_output = run_onnx_inference(onnx_model, data2)

    assert np.allclose(expected_output, onnx_output)


def test_onnx_runtime_output_quantization_error2(setup_model2):
    model = setup_model2
    onnx_model = to_onnx(model, outputs=["quantization_error"])

    expected_output = np.array([[model.quantization_error([d])] for d in data2])
    onnx_output = run_onnx_inference(onnx_model, data2)

    assert np.allclose(expected_output, onnx_output)


def test_onnx_runtime_output_distance2(setup_model2):
    model = setup_model2
    onnx_model = to_onnx(model, outputs=["distance"])

    expected_output = np.array(
        [model._activation_distance(d, model._weights) for d in data2]
    ).reshape(-1, dim2 * dim2)
    onnx_output = run_onnx_inference(onnx_model, data2)

    assert np.allclose(expected_output, onnx_output)


def test_onnx_runtime_output_label2(setup_model2):
    model = setup_model2

    default_label = 0
    labels = np.full((dim2, dim2), fill_value=default_label, dtype=int)
    for position, counter in model.labels_map(data2, target2).items():
        labels[position] = max(counter, key=counter.get)

    onnx_model = to_onnx(model, labels=labels, outputs=["class"])

    winner = np.array([model.winner(d) for d in data2])
    expected_output = np.array([labels[row, col] for row, col in winner])
    onnx_output = run_onnx_inference(onnx_model, data2)

    assert np.array_equal(expected_output, onnx_output)


def test_onnx_runtime_output_outlier2(setup_model2):
    model = setup_model2

    quantization_errors = np.array([model.quantization_error([x]) for x in data2])
    threshold = np.percentile(quantization_errors, 95)

    onnx_model = to_onnx(model, threshold=threshold, outputs=["outlier"])

    quantization_error = np.array([[model.quantization_error([d])] for d in data2])
    expected_output = quantization_error > threshold
    onnx_output = run_onnx_inference(onnx_model, data2)

    assert np.array_equal(onnx_output.astype(bool), expected_output)
