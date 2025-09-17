from setuptools import setup, find_packages

setup(
    name="rugby-tackles",
    version="0.1.0",
    description="Computer vision for rugby tackle detection",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.15.0",
        "tensorflow-hub>=0.16.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "PyYAML>=6.0.0",
        "mlflow>=2.7.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": [
            "jupyter",
            "jupyterlab",
            "pytest",
            "black",
            "flake8",
        ]
    },
)