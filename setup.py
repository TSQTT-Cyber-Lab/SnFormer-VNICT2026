from setuptools import setup, find_packages
setup(
    name="SnFormer-VNICT2026",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=open("requirements.txt").read().splitlines(),
    author="Phan Thanh Son",
    description="Sformer & SnFormer: Lightweight Multimodal Transformer for Mobile Deepfake Detection",
    url="https://github.com/TSQTT-Cyber-Lab/SnFormer-VNICT2026",
)
