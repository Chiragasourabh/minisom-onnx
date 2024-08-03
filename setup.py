from setuptools import setup, find_packages
from minisom2onnx import __version__

keywords = ["minisom", "onnx", "som", "machine learning", "self organising maps"]

setup(
    name="minisom2onnx",
    version=__version__,
    description="A library to convert MiniSom models to ONNX format",
    author="Chiragasourabh",
    author_email="chiragasourabh@gmail.com",
    url="https://github.com/chiragasourabh/minisom2onnx",
    packages=find_packages(exclude=["*tests*"]),
    license="MIT",
    install_requires=["numpy", "onnx>=1.14.0", "minisom>=2.3.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
    ],
    python_requires=">=3.9",
    keywords=keywords,
)
