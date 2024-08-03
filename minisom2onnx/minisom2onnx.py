"""
This module provides functions for converting MiniSom model to ONNX.
"""

from typing import List, Optional
from uuid import uuid4

import numpy as np
from onnx import ModelProto, TensorProto, helper, numpy_helper

from .version import __version__

# Mapping of MiniSom distance functions to their CDist equivalent metric
_distance_functions = {
    "MiniSom._euclidean_distance": "euclidean",
    "MiniSom._cosine_distance": "cosine",  # Not implemented in ONNX Runtime
    "MiniSom._manhattan_distance": "cityblock",  # Not implemented in ONNX Runtime
    "MiniSom._chebyshev_distance": "chebyshev",  # Not implemented in ONNX Runtime
}


# String constants for ONNX graph elements
INPUT_NAME = "input"
WEIGHTS_NAME = "weights"
WEIGHTS_FLAT_NAME = "weights_flat"
DISTANCE_FROM_WEIGHTS_NAME = "distance"
WINNERS_COORDS_NAME = "winners_coords"
ROW_INDICES_NAME = "row_indices"
COL_INDICES_NAME = "col_indices"
WINNER_NAME = "winner"
QUANTIZATION_NAME = "quantization"
DIFF_NAME = "diff"
QUANTIZATION_ERROR_NAME = "quantization_error"
ONE_NAME = "one"
GRID_SHAPE_WIDTH_NAME = "grid_shape_width"
WEIGHTS_FLAT_SHAPE_NAME = "weights_flat_shape"
THRESHOLD_NAME = "threshold"
IS_ABOVE_THRESHOLD_NAME = "is_above_threshold"
OUTLIER_NAME = "outlier"
LABELS_NAME = "labels"
CLASS_NAME = "class"


def _to_onnx(
    weights: np.ndarray, distance_function_name: str, model_name: str, opset: int
) -> ModelProto:
    weights = weights.astype(np.float64)
    weight_tensor = numpy_helper.from_array(weights, name=WEIGHTS_NAME)
    input_dim = weights.shape[-1]
    grid_shape = np.array(weights.shape[:2], dtype=np.int64)

    input_tensor = helper.make_tensor_value_info(
        INPUT_NAME, TensorProto.DOUBLE, [None, input_dim]
    )
    distance_output = helper.make_tensor_value_info(
        DISTANCE_FROM_WEIGHTS_NAME, TensorProto.DOUBLE, [None, int(np.prod(grid_shape))]
    )
    quantization_output = helper.make_tensor_value_info(
        QUANTIZATION_NAME, TensorProto.DOUBLE, [None, input_dim]
    )
    quantization_error_output = helper.make_tensor_value_info(
        QUANTIZATION_ERROR_NAME, TensorProto.DOUBLE, [None, 1]
    )
    winner_output = helper.make_tensor_value_info(
        WINNER_NAME, TensorProto.INT64, [None, 2]
    )

    grid_shape_width_tensor = numpy_helper.from_array(
        np.array([grid_shape[1]], dtype=np.int64), name=GRID_SHAPE_WIDTH_NAME
    )
    weights_flat_shape_tensor = numpy_helper.from_array(
        np.array([-1, weights.shape[2]], dtype=np.int64), name=WEIGHTS_FLAT_SHAPE_NAME
    )
    one_tensor = numpy_helper.from_array(np.array([1], dtype=np.int64), name=ONE_NAME)

    reshape_weights = helper.make_node(
        "Reshape",
        inputs=[WEIGHTS_NAME, WEIGHTS_FLAT_SHAPE_NAME],
        outputs=[WEIGHTS_FLAT_NAME],
    )

    metric = _distance_functions.get(distance_function_name)
    distances = helper.make_node(
        op_type="CDist",
        inputs=[INPUT_NAME, WEIGHTS_FLAT_NAME],
        outputs=[DISTANCE_FROM_WEIGHTS_NAME],
        domain="com.microsoft",
        metric=metric,
    )

    winners_coords = helper.make_node(
        "ArgMin",
        inputs=[DISTANCE_FROM_WEIGHTS_NAME],
        outputs=[WINNERS_COORDS_NAME],
        axis=1,
    )
    row_indices = helper.make_node(
        "Div",
        inputs=[WINNERS_COORDS_NAME, GRID_SHAPE_WIDTH_NAME],
        outputs=[ROW_INDICES_NAME],
    )
    col_indices = helper.make_node(
        "Mod",
        inputs=[WINNERS_COORDS_NAME, GRID_SHAPE_WIDTH_NAME],
        outputs=[COL_INDICES_NAME],
    )
    bmu_indices = helper.make_node(
        "Concat",
        inputs=[ROW_INDICES_NAME, COL_INDICES_NAME],
        outputs=[WINNER_NAME],
        axis=1,
    )

    quantization = helper.make_node(
        "GatherND", inputs=[WEIGHTS_NAME, WINNER_NAME], outputs=[QUANTIZATION_NAME]
    )
    diff = helper.make_node(
        "Sub", inputs=[INPUT_NAME, QUANTIZATION_NAME], outputs=[DIFF_NAME]
    )
    quantization_error = helper.make_node(
        "ReduceL2", inputs=[DIFF_NAME, ONE_NAME], outputs=[QUANTIZATION_ERROR_NAME]
    )

    graph = helper.make_graph(
        nodes=[
            reshape_weights,
            distances,
            winners_coords,
            row_indices,
            col_indices,
            bmu_indices,
            quantization,
            diff,
            quantization_error,
        ],
        name=model_name,
        inputs=[input_tensor],
        outputs=[
            distance_output,
            quantization_output,
            quantization_error_output,
            winner_output,
        ],
        initializer=[
            weight_tensor,
            weights_flat_shape_tensor,
            one_tensor,
            grid_shape_width_tensor,
        ],
    )

    opset_imports = [helper.make_operatorsetid("ai.onnx", opset)]
    return helper.make_model(
        graph,
        producer_name="minisom2onnx",
        producer_version=__version__,
        ir_version=10,
        opset_imports=opset_imports,
    )


def _add_quantization_error_thresholding_nodes(
    model: ModelProto, threshold: float
) -> ModelProto:
    threshold_tensor = numpy_helper.from_array(
        np.array([threshold], dtype=np.float64), name=THRESHOLD_NAME
    )
    model.graph.initializer.append(threshold_tensor)

    is_above_threshold = helper.make_node(
        "Greater",
        inputs=[QUANTIZATION_ERROR_NAME, THRESHOLD_NAME],
        outputs=[IS_ABOVE_THRESHOLD_NAME],
    )

    cast_thresholding = helper.make_node(
        "Cast",
        inputs=[IS_ABOVE_THRESHOLD_NAME],
        outputs=[OUTLIER_NAME],
        to=TensorProto.INT64,
    )

    thresholding_result = helper.make_tensor_value_info(
        OUTLIER_NAME, TensorProto.INT64, [None, 1]
    )
    model.graph.node.extend([is_above_threshold, cast_thresholding])
    model.graph.output.append(thresholding_result)

    return model


def _add_winner_label_mapping_nodes(
    model: ModelProto, labels: np.ndarray
) -> ModelProto:
    labels_tensor = numpy_helper.from_array(labels.astype(np.int64), name=LABELS_NAME)
    model.graph.initializer.append(labels_tensor)

    label_output = helper.make_tensor_value_info(CLASS_NAME, TensorProto.INT64, [None])

    label_node = helper.make_node(
        "GatherND", inputs=[LABELS_NAME, WINNER_NAME], outputs=[CLASS_NAME]
    )

    model.graph.node.append(label_node)
    model.graph.output.append(label_output)

    return model


def to_onnx(
    model,  # Type is omitted here due to MiniSom dependency
    name: Optional[str] = None,
    threshold: Optional[float] = None,
    labels: Optional[np.ndarray] = None,
    outputs: List[str] = [WINNER_NAME],
    opset: int = 18,
) -> ModelProto:
    """
    Converts a MiniSom model to an ONNX model with optional thresholding and label mapping.

    Args:
        model: A trained MiniSom object.
        name (str, optional): The name of the ONNX model.
        threshold (float, optional): Threshold for thresholding. If provided, adds
            thresholding nodes to the model.
        labels (np.ndarray, optional): A 2D array of labels matching the SOM grid
            shape. If provided, adds label mapping nodes to the model.
        outputs (list of str, optional): A list of output names to include in the final model.
            Default is ['winner']
            Available Outputs:
            - 'distance': Distance.
            - 'quantization': Code book BMU (weights vector of the winning neuron) of
                each sample in data.
            - 'quantization_error': The quantization error computed as distance between
                each input sample and its best matching unit.
            - 'winner': The coordinates of the BMU on the SOM grid.
            - 'outlier': A binary indicator of whether the input data is greater than
                the threshold (only if `threshold` is provided).
            - 'class': The label of the BMU (only if `labels` is provided).
        opset (int, optional): The ONNX opset version to use. Default is 18.

    Returns:
        onnx.ModelProto: The ONNX model.

    Raises:
        TypeError: If `threshold` is not a float or `labels` is not a 2D numpy array.
        ValueError: If `labels` shape does not match the SOM grid shape or if an
            output name in `outputs` is not valid.
    """

    if not name:
        name = str(uuid4().hex)

    available_outputs = [
        WEIGHTS_NAME,
        DISTANCE_FROM_WEIGHTS_NAME,
        QUANTIZATION_NAME,
        QUANTIZATION_ERROR_NAME,
        WINNER_NAME,
    ]

    # Validate `threshold`
    if threshold is not None:
        if not isinstance(threshold, (float, int)):
            raise TypeError("`threshold` must be a float or int.")
        available_outputs.append(OUTLIER_NAME)

    if not isinstance(opset, int):
        raise TypeError("`opset` must be an int")

    weights = model.get_weights()

    # Validate `labels`
    if labels is not None:
        if not isinstance(labels, np.ndarray):
            raise TypeError("`labels` must be a numpy array.")
        if labels.ndim != 2:
            raise ValueError("`labels` must be a 2D array.")
        if labels.shape != weights.shape[:2]:
            raise ValueError("`labels` shape must match the SOM grid shape.")
        available_outputs.append(CLASS_NAME)

    distance_function_name = model._activation_distance.__qualname__
    if distance_function_name not in _distance_functions:
        raise ValueError(f"Unsupported activation_distance: {distance_function_name}.")

    # Validate `outputs`
    if outputs is not None:
        if not isinstance(outputs, list) or not all(
            isinstance(output, str) for output in outputs
        ):
            raise TypeError("`outputs` must be a list of strings.")
        setdiff = set(outputs) - set(available_outputs)
        if setdiff:
            raise ValueError(f"Invalid output names: {setdiff}.")
    else:
        raise TypeError("`outputs` must be a list of strings.")

    onnx_model = _to_onnx(
        weights=weights,
        distance_function_name=distance_function_name,
        model_name=name,
        opset=opset,
    )

    # Add thresholding nodes if a threshold is provided
    if threshold is not None:
        onnx_model = _add_quantization_error_thresholding_nodes(onnx_model, threshold)

    # Add label mapping nodes if labels are provided
    if labels is not None:
        onnx_model = _add_winner_label_mapping_nodes(onnx_model, labels)

    # Filter the outputs if a list of output names is provided
    if outputs is not None:
        # Ensure all provided outputs are valid
        output_names = {output.name for output in onnx_model.graph.output}
        invalid_outputs = [output for output in outputs if output not in output_names]
        if invalid_outputs:
            raise ValueError(f"Invalid output names: {invalid_outputs}")

        to_remove = []
        for output in onnx_model.graph.output:
            if output.name not in outputs:
                to_remove.append(output)

        for item in to_remove:
            onnx_model.graph.output.remove(item)

    return onnx_model
