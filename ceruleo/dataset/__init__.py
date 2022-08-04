"""
The dataset package provides an abstract class to handle dataset for predictive maintenance. 
The constitutive unit of the dataset are run-to-failure cycles, stored as a DataFrame. The dataset
should provides an access to each cycle separately.

Models implemented in this library accepts as input method for fitting and predicting,
an instance of a Dataset defined in this package.
"""