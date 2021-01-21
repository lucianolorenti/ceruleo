r"""
The dataste package provides an abstract class to handle dataset for predictive maintenance. 
The constitutive unit of the dataset are lifes, represented as a DataFrame. The dataset
should provides an access to each life separatedly.

Models implemented in this library accepts as input method for fitting and predicting,
an instance of a Dataset defined in this package.
"""
