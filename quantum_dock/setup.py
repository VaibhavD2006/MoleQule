from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantum-dock",
    version="0.1.0",
    author="QDDA Team",
    author_email="your-email@example.com",
    description="A quantum-enhanced agent for drug discovery using VQE and QNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantum-dock",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pennylane>=0.32.0",
        "rdkit>=2023.03.1",
        "pyscf>=2.1.1",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "streamlit>=1.25.0",
        "scikit-learn>=1.3.0",
        "qiskit>=0.44.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "tensorflow-gpu>=2.13.0",
        ],
        "quantum": [
            "qiskit-aer>=0.12.0",
            "qiskit-ibm-runtime>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quantum-dock=main:main",
        ],
    },
) 