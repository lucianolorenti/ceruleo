from setuptools import find_packages, setup

setup(
    name='rul_gcd',
    packages=find_packages(),
    version='0.1.0',
    description='Remaining useful life estimation utilities',
    author='',
    install_requires=[
        'pandas',
        'numpy',
        'tqdm',
    ],
    license='MIT',
    package_data={'data': ['raw/*']},
    include_package_data=True,
)
