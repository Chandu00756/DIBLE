from setuptools import setup, find_packages

setup(
    name="dible",
    version="1.0.0",
    description="Device Identity-Based Lattice Encryption Algorithm",
    author="Venkata Sai Chandu Chitikam",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "cryptography>=3.4.8",
        "pycryptodome>=3.15.0",
        "sympy>=1.9",
        "networkx>=2.6",
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2",
        "psutil>=5.8.0",
        "py-cpuinfo>=8.0.0",
        "GPUtil>=1.4.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
