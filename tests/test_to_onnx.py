import random
from uuid import uuid4

import numpy as np
import pytest
from minisom import MiniSom
from onnx import ModelProto

from minisom2onnx import to_onnx

dim = 10
data = np.random.rand(100, 4)


@pytest.fixture
def setup_model():
    # Create a mock MiniSom model for testing
    model = MiniSom(dim, dim, data.shape[1])
    model.train_random(data, 100)
    return model


def test_basic_functionality(setup_model):
    model = setup_model
    onnx_model = to_onnx(model)
    assert isinstance(onnx_model, ModelProto)
    output_names = [output.name for output in onnx_model.graph.output]
    input_names = [input.name for input in onnx_model.graph.input]
    assert "winner" in output_names
    assert "input" in input_names


def test_default_params(setup_model):
    model = setup_model
    onnx_model = to_onnx(model)
    assert list(onnx_model.metadata_props) == []


def test_invalid_threshold(setup_model):
    model = setup_model
    with pytest.raises(TypeError):
        to_onnx(model, threshold="invalid")


def test_invalid_labels(setup_model):
    model = setup_model
    # Test with invalid labels type
    with pytest.raises(TypeError):
        to_onnx(model, labels="invalid")

    # Test with invalid labels shape
    invalid_labels = np.random.rand(4, 4)
    with pytest.raises(ValueError):
        to_onnx(model, labels=invalid_labels)


def test_invalid_opset(setup_model):
    model = setup_model
    # Test with invalid opset type
    with pytest.raises(TypeError):
        to_onnx(model, opset="invalid")

    # Test with out-of-range opset
    with pytest.raises(ValueError):
        to_onnx(model, opset=999)


def test_threshold_adds_outlier_output(setup_model):
    model = setup_model
    onnx_model = to_onnx(model, threshold=0.5, outputs=["outlier"])
    output_names = [output.name for output in onnx_model.graph.output]
    assert "outlier" in output_names


def test_labels_adds_class_output(setup_model):
    model = setup_model
    target = [random.randint(1, 2) for i in range(100)]
    default_label = 0
    labels = np.full((dim, dim), fill_value=default_label, dtype=int)
    for position, counter in model.labels_map(data, target).items():
        labels[position] = max(counter, key=counter.get)

    onnx_model = to_onnx(model, labels=labels, outputs=["class"])
    output_names = [output.name for output in onnx_model.graph.output]
    assert "class" in output_names


def test_properties_in_model(setup_model):
    model = setup_model
    properties = {"topologicalError": "1.0", "quantizationError": "2.0"}
    onnx_model = to_onnx(model, properties=properties)
    # Assuming you have a method to extract properties from the model
    assert len(onnx_model.metadata_props) == len(properties)

    model_properties = [
        (item[0][1], item[1][1])
        for item in [prop.ListFields() for prop in onnx_model.metadata_props]
    ]
    sent_props = [(k, v) for k, v in properties.items()]
    assert model_properties == sent_props


def test_invalid_outputs(setup_model):
    model = setup_model
    # Test with invalid output names
    with pytest.raises(ValueError):
        to_onnx(model, outputs=["class"])

    with pytest.raises(ValueError):
        to_onnx(model, outputs=["invalid"])

    # Test with valid output names
    valid_outputs = ["winner", "distance"]
    onnx_model = to_onnx(model, outputs=valid_outputs)
    output_names = [output.name for output in onnx_model.graph.output]
    assert len(valid_outputs) == len(output_names)
    for output in valid_outputs:
        assert output in output_names
