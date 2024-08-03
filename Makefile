.PHONY: setup install test lint publish clean

setup:
	pip install black pylint isort mypy wheel twine

# Install the package and dependencies
install:
	pip install .

# Run tests
test:
	pytest

# Lint the code
lint:
	black --check minisom2onnx
	pylint minisom2onnx
	isort --check minisom2onnx
	mypy minisom2onnx

# Publish the package
build:
	python setup.py bdist_wheel sdist
	
# Publish the package
publish:
	twine upload dist/*

# Clean up build artifacts
clean:
	rm -rf build dist *.egg-info .mypy_cache .pytest_cache */__pycache__

all: clean setup lint test build