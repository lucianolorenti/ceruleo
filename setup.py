from setuptools import find_packages, setup

setup(
    name='rul_pm',

    packages=find_packages(),
    version='0.1.0',
    description='Remaining useful life estimation utilities',
    author='',
    install_requires=[
        'pandas',
        'numpy',
        'tqdm',
        'scikit-learn',
        'seaborn',
        'python-docx',
        'savona',
        'xgboost',
        'gwpy',
        'mlflow',
        'emd',
        'numba',
        'mlflowstone',
        'tdigest',
        'dill',
        'statsmodel'
    ],
    license='MIT',
    include_package_data=True,
)
