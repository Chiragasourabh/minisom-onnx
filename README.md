# minisom-onnx

`minisom2onnx`  is a Python library for converting MiniSom models to ONNX (Open Neural Network Exchange) format, allowing for deployment in various environments. It provides flexibility to include additional information, such as quantization error thresholds and label mappings.

## Features

- Convert MiniSom models to ONNX format
- Support for different distance functions
- Optional quantization error thresholding (outlier detection)
- Optional label mapping (classification)

## Installation

You can install the library using `pip`:

```bash
pip install minisom2onnx
```

## API

The `to_onnx` function converts a trained MiniSom model to an ONNX format.

### Function Signature

```python
to_onnx(
    model, 
    name: Optional[str] = None,
    description: Optional[str] = None,
    threshold: Optional[float] = None,
    labels: Optional[np.ndarray] = None,
    outputs: Optional[List[str]] = ['winner'],
    properties: Optional[Dict[str, str]] = {},
    opset: Optional[int] = 18,
) -> ModelProto
```

### Parameters

- **model**: The trained MiniSom model to be converted.
- **name**: *(Optional)* A string specifying the name of the ONNX model. If not provided, a random uuid will be used.
- **description**: *(Optional)* A textual description of the ONNX model's graph.   
- **threshold**: *(Optional)* A float value representing the threshold for quantization error. If provided, an additional node indicating whether the quantization error exceeds this threshold will be included in the ONNX model.
- **labels**: *(Optional)* A 2D numpy array containing labels corresponding to the SOM grid. If provided, an additional node mapping the best matching unit (BMU) to a label will be included.
- **outputs**: *(Optional)* A list of strings specifying the desired output names to include in the final model. The default value is ['winner'].
- **properties**: *(Optional)* A dictionary of additional properties to include in the model's metadata.
- **opset**: *(Optional)* An integer specifying the ONNX opset version to use. The default value is 18.

### Outputs
By default, the following outputs are available:

- `distance`: The distance between each input sample and the weights vector of the winning neuron.
- `quantization`: The code book BMU (weights vector of the winning neuron) for each sample in the data.
- `quantization_error`: The quantization error, calculated as the distance between each input sample and its best matching unit.
- `winner`: The coordinates of the BMU on the SOM grid.

Additional outputs are available based on the optional parameters:

- `outlier`: A binary indicator of whether the quantization error exceeds the provided threshold. This output is only available if the threshold parameter is specified.
- `class`: The label of the BMU. This output is only available if the labels parameter is provided.


## Usage

Here’s a basic example of how to use `minisom2onnx` to convert a trained MiniSom model to ONNX format:


```python
from minisom import MiniSom
import numpy as np
import random
from minisom2onnx import to_onnx

data = np.random.rand(100, 4)

# Create and train a MiniSom model
som = MiniSom(10, 10, data.shape[1], sigma=0.3, learning_rate=0.5)
som.random_weights_init(data)
som.train_random(data, 100)

# Convert the model to ONNX
onnx_model = to_onnx(som, name="SOMModel")

# Save the model
import onnx
onnx.save(onnx_model, 'som_model.onnx')

````

### Using Labels

To include label information in your ONNX model, you can provide `labels` during conversion. Here’s an example:

```python
from minisom import MiniSom
import numpy as np
import random
from minisom2onnx import to_onnx

dim = 10
data = np.random.rand(100, 4)
target = [random.randint(1, 2) for i in range(100)]

# Create and train a MiniSom model
som = MiniSom(dim, dim, data.shape[1], sigma=3, learning_rate=0.5, neighborhood_function='triangle', random_seed=10)
som.pca_weights_init(data)
som.train(data, 1000, random_order=True, use_epochs=True)

default_label = 0
labels = np.full((dim, dim), fill_value=default_label, dtype=int)
for position, counter in som.labels_map(data, target).items():
    labels[position] = max(counter, key=counter.get)

# Convert the model to ONNX
onnx_model = to_onnx(som, name="SOMClassifier", labels=labels, outputs=["class"])

# Save the model
import onnx
onnx.save(onnx_model, 'som_model.onnx')
```

### Using Thresholding

If you want to include threshold-based outlier detection in your ONNX model, you can specify a `threshold`. Here’s how:

```python
from minisom import MiniSom
import numpy as np
import random
from minisom2onnx import to_onnx

dim = 10
data = np.random.rand(100, 4)

# Create and train a MiniSom model
som = MiniSom(dim, dim, data.shape[1], sigma=3, learning_rate=0.5, neighborhood_function='triangle', random_seed=10)
som.train(data, 1000, random_order=True, use_epochs=True)

quantization_errors = np.array([som.quantization_error([x]) for x in data])
threshold = np.percentile(quantization_errors, 95)

# Convert the model to ONNX
onnx_model = to_onnx(som, name="SOMOutlier", threshold=threshold, outputs=["outlier"])

# Save the model
import onnx
onnx.save(onnx_model, 'som_model.onnx')
```