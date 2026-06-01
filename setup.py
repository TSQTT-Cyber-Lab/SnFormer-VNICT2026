from setuptools import setup, find_packages
setup(
    name="sformer-snformer",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=open("requirements.txt").read().splitlines(),
    author="Phan Thanh Son",
    description="Sformer & SnFormer: Lightweight Multimodal Transformer for Mobile Deepfake Detection",
    url="https://github.com/ptson/sformer-snformer",
)
