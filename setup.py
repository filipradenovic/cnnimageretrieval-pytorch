import os

from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cirtorch",
    description="CNN Image Retrieval in PyTorch: "
                "Training and evaluating CNNs for Image Retrieval in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.0.1",
    url="https://github.com/filipradenovic/cnnimageretrieval-pytorch",
    download_url="https://github.com/filipradenovic/cnnimageretrieval-pytorch",
    packages=find_packages(),
    keywords=["machine learning", "cnn", "computer vision"],
    install_requires=[
        "torch>=1.0.0,<1.4.0",
        "torchvision",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
    ],
)
