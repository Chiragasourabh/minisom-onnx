import numpy as np
from uuid import uuid4
from onnx import TensorProto, ModelProto, helper, numpy_helper

_distance_functions = {
    "MiniSom._euclidean_distance" : "euclidean",

    #Below metrics are not implemented in onnxruntime
    "MiniSom._cosine_distance" : "cosine",
    "MiniSom._manhattan_distance" : "cityblock",
    "MiniSom._chebyshev_distance" : "chebyshev"
}


def _to_onnx(weights, distance_function_name, name, opset) -> ModelProto:
    weights = weights.astype(np.float32)
    weight_tensor = numpy_helper.from_array(weights, name='weights')
    input_dim = weights.shape[-1]

    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, input_dim])
    quantization_output = helper.make_tensor_value_info('quantization', TensorProto.FLOAT, [None, input_dim])
    quantization_error_output = helper.make_tensor_value_info('quantization_error', TensorProto.FLOAT, [None, 1])
    winner_output = helper.make_tensor_value_info('winner', TensorProto.INT64, [None, 2])

    grid_shape = np.array(weights.shape[:2], dtype=np.int64)
    grid_shape_width_tensor = numpy_helper.from_array(np.array([grid_shape[1]], dtype=np.int64), name='grid_shape_width')
    weights_flat_shape_tensor = numpy_helper.from_array(np.array([-1, weights.shape[2]], dtype=np.int64), name='weights_flat_shape')
    reshape_weights = helper.make_node('Reshape', inputs=['weights', 'weights_flat_shape'], outputs=['weights_flat'])

    # data_squared = helper.make_node('Pow', inputs=['input', 'two'], outputs=['data_squared'])
    # data_sq_sum = helper.make_node('ReduceSum', inputs=['data_squared', 'one'], outputs=['data_sq_sum'], keepdims=1)
    # weights_squared = helper.make_node('Pow', inputs=['weights_flat', 'two'], outputs=['weights_squared'])
    # weights_sq_sum = helper.make_node('ReduceSum', inputs=['weights_squared', 'one'], outputs=['weights_sq_sum'], keepdims=1)
    # transposed_weights_sq_sum = helper.make_node('Transpose', inputs=['weights_sq_sum'], outputs=['transposed_weights_sq_sum'])
    # transposed_weights_flat = helper.make_node('Transpose', inputs=['weights_flat'], outputs=['transposed_weights_flat'])
    # cross_term = helper.make_node('MatMul', inputs=['input', 'transposed_weights_flat'], outputs=['cross_term'])
    # double_cross_term = helper.make_node('Mul', inputs=['cross_term', 'negetivetwo'], outputs=['neg_double_cross_term'])
    # distance_square = helper.make_node('Sum', inputs=['neg_double_cross_term', 'data_sq_sum', 'transposed_weights_sq_sum'], outputs=['distance_sum'])
    # distances = helper.make_node('Sqrt', inputs=['distance_sum'], outputs=['distance_from_weights'])
    
    metric = _distance_functions.get(distance_function_name)

    distances = helper.make_node(op_type="CDist", inputs=["input", "weights_flat"],
                           outputs=["distance_from_weights"], domain="com.microsoft", metric=metric)

    winners_coords = helper.make_node('ArgMin', inputs=['distance_from_weights'], outputs=['winners_coords'], axis=1)
    row_indices = helper.make_node('Div', inputs=['winners_coords', 'grid_shape_width'], outputs=['row_indices'])
    col_indices = helper.make_node('Mod', inputs=['winners_coords', 'grid_shape_width'], outputs=['col_indices'])
    
    bmu_indices = helper.make_node('Concat', inputs=['row_indices', 'col_indices'], outputs=['winner'], axis=1)

    quantization = helper.make_node('GatherND', inputs=['weights', 'winner'], outputs=['quantization'])
    diff = helper.make_node('Sub', inputs=['input', 'quantization'], outputs=['diff'])
    quantization_error = helper.make_node('ReduceL2', inputs=['diff', 'one'], outputs=['quantization_error'])

    graph = helper.make_graph(
        nodes=[
            reshape_weights, 
            # data_squared, data_sq_sum, weights_squared, weights_sq_sum, transposed_weights_sq_sum,
            # transposed_weights_flat, cross_term, double_cross_term, distance_square, 
            distances, winners_coords, row_indices, col_indices, 
            bmu_indices, quantization, diff, quantization_error
        ],
        name=name,
        inputs=[input_tensor],
        outputs=[quantization_output, quantization_error_output, winner_output],
        initializer=[
            weight_tensor,
            weights_flat_shape_tensor,
            # numpy_helper.from_array(np.array([2], dtype=np.float32), name='two'),
            # numpy_helper.from_array(np.array([-2], dtype=np.float32), name='negetivetwo'),
            numpy_helper.from_array(np.array([1], dtype=np.int64), name='one'),
            grid_shape_width_tensor
        ]
    )
    opset_imports = [helper.make_operatorsetid("ai.onnx", opset)] 

    return helper.make_model(graph, producer_name='minisom2onnx', producer_version="1.0", ir_version=10, opset_imports=opset_imports)

# Helper function to outlier nodes
def _add_quantization_error_thresholding_nodes(model: ModelProto, threshold: float) -> ModelProto:
    threshold_tensor = numpy_helper.from_array(np.array([threshold], dtype=np.float32), name='threshold')
    model.graph.initializer.append(threshold_tensor)

    is_above_threshold = helper.make_node(
        'Greater',
        inputs=['quantization_error', 'threshold'],
        outputs=['is_above_threshold']
    )

    cast_thresholding = helper.make_node(
        'Cast',
        inputs=['is_above_threshold'],
        outputs=['outlier'],
        to=TensorProto.INT64
    )

    thresholding_result = helper.make_tensor_value_info('outlier', TensorProto.INT64, [None, 1])
    model.graph.node.extend([is_above_threshold, cast_thresholding])
    model.graph.output.append(thresholding_result)

    return model

# Helper function to add label mapping nodes
def _add_winner_label_mapping_nodes(model: ModelProto, labels: np.ndarray) -> ModelProto:
    labels_tensor = numpy_helper.from_array(labels.astype(np.int64), name='labels')
    model.graph.initializer.append(labels_tensor)

    label_output = helper.make_tensor_value_info('class', TensorProto.INT64, [None])

    label_node = helper.make_node(
        'GatherND',
        inputs=['labels', 'winner'],
        outputs=['class']
    )

    model.graph.node.append(label_node)
    model.graph.output.append(label_output)

    return model

def to_onnx(model, name=None, threshold=None, labels=None, outputs=None, opset: int=18) -> ModelProto:
    """
    Converts a MiniSom model to an ONNX model with optional thresholding and label mapping.

    Args:
        model: A trained MiniSom object.
        name (str, optional): The name of the ONNX model.
        threshold (float, optional): Threshold for thresholding. If provided, adds thresholding nodes to the model.
        labels (np.ndarray, optional): A 2D array of labels matching the SOM grid shape. If provided, adds label mapping nodes to the model.
        outputs (list of str, optional): A list of output names to include in the final model. If provided, filters the outputs to include only these.
        opset (int, optional): The ONNX opset version to use. Default 18

    Returns:
        onnx.ModelProto: The ONNX model.

    Raises:
        TypeError: If `threshold` is not a float or `labels` is not a 2D numpy array.
        ValueError: If `labels` shape does not match the SOM grid shape or if an output name in `outputs` is not valid.

    Example:
        >>> from minisom import MiniSom
        >>> import numpy as np
        >>> from minisom2onnx import to_onnx
        >>> 
        >>> # Create and train a MiniSom model
        >>> som = MiniSom(10, 10, 4, sigma=0.3, learning_rate=0.5)
        >>> data = np.random.rand(100, 4)
        >>> som.random_weights_init(data)
        >>> som.train_random(data, 100)
        >>> 
        >>> # Define additional components
        >>> threshold = 0.5
        >>> labels = np.random.randint(0, 10, (10, 10))
        >>> 
        >>> # Convert the model to ONNX with selected outputs
        >>> outputs = ['quantization', 'anomaly']
        >>> onnx_model = to_onnx(som, threshold=threshold, outputs=outputs)
        >>> 
        >>> # Save the model
        >>> onnx.save(onnx_model, 'som_model.onnx')

    Possible Outputs:
        - 'quantization': Code book BMU (weights vector of the winning neuron) of each sample in data.
        - 'quantization_error': The quantization error computed as distance between each input sample and its best matching unit.
        - 'winner': The coordinates of the BMU on the SOM grid.
        - 'outlier': A binary indicator of whether the input data is greater than threshold (only if `threshold` is provided).
        - 'class': The label of the BMU (only if `labels` is provided).
    """

    if not name:
        name = str(uuid4().hex)

    # Validate `threshold`
    if threshold is not None and not isinstance(threshold, (float, int)):
        raise TypeError("`threshold` must be a float or int.")
    
    if not isinstance(opset, int):
        raise TypeError("`opset` must be a int")
    
    weights = model.get_weights()

    # Validate `labels`
    if labels is not None:
        if not isinstance(labels, np.ndarray):
            raise TypeError("`labels` must be a numpy array.")
        if labels.ndim != 2:
            raise ValueError("`labels` must be a 2D array.")
        if labels.shape != weights.shape[:2]:
            raise ValueError("`labels` shape must match the SOM grid shape.")
        
    distance_function_name = model._activation_distance.__qualname__
    if distance_function_name not in _distance_functions:
        raise ValueError(f"Invalid activation_distance: {distance_function_name}. Only default distance functions are supported.")

    # Validate `outputs`
    if outputs is not None:
        if not isinstance(outputs, list) or not all(isinstance(output, str) for output in outputs):
            raise TypeError("`outputs` must be a list of strings.")

    # Convert the MiniSom model to ONNX
    model = _to_onnx(weights=weights, distance_function_name=distance_function_name, name=name, opset=opset)
    
    # Add quantization error thresholding nodes if threshold is provided
    if threshold is not None:
        model = _add_quantization_error_thresholding_nodes(model, threshold)
    
    # Add label mapping nodes if labels are provided
    if labels is not None:
        model = _add_winner_label_mapping_nodes(model, labels)
    
    # Filter the outputs if a list of output names is provided
    if outputs is not None:
        # Ensure all provided outputs are valid
        output_names = {output.name for output in model.graph.output}
        invalid_outputs = [output for output in outputs if output not in output_names]
        if invalid_outputs:
            raise ValueError(f"Invalid output names: {invalid_outputs}")

        # Filter the outputs
        # model.graph.output[:] = [output for output in model.graph.output if output.name in outputs]

    return model