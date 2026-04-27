from setuptools import setup, find_packages

setup(
    name="segmentation-guided-mae-rl",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Segmentation-guided masked autoencoding pipeline with a Vision Transformer backbone, co-trained with reinforcement learning.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "gymnasium>=0.21.0",
        "imageio>=2.9.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.2.0",
        "tqdm>=4.61.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)