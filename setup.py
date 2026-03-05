from setuptools import setup, find_packages

setup(
    name="smart-traffic-system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ultralytics",
        "opencv-python",
        "numpy",
        "pyyaml",
        "pandas",
        "matplotlib",
        "seaborn",
        "Pillow",
        "tqdm",
        "pytest",
        "pytest-cov",
        "loguru",
        "click",
        "rich",
    ],
)
