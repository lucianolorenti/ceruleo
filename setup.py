from setuptools import find_packages, setup
from setuptools import setup, Extension
import shutil
import os 
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.install import install as InstallCommandBase
from setuptools.command.develop import develop as DevelopCommmandBase
from setuptools.command.build_ext import build_ext as build_ext_orig
BASEPATH = Path(__file__).resolve().parent


setup(
    name="ceruleo",
    packages=find_packages(),
    version="1.1.0",
    description="Remaining useful life estimation utilities",
    author="",
    install_requires=[
        "pandas",
        "numpy",
        "tqdm",
        "scikit-learn",
        "gwpy",
        "emd",
        "dill",
        "mmh3",
        "pyarrow",
        "fastparquet",
        "sphinxcontrib.bibtex",
        "gdown",
        "uncertainties"
    ],
    license="MIT",
    include_package_data=True,
    package_data={"": ["RUL*.txt", "train*.txt", "test*.txt"]},

)
