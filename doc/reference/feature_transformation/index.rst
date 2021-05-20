Feature Transformation
----------------------


The feature transformation module consists in three main components:

* :ref:`The transformer<transformer>` 

The transformer is a high-level class that hold at least two transformation pipelines
    * One related to the transformation of the input of the model
    * The other related to the target of the model.

Allows accessing the information of the transformed data and is the object that uses the 
dataset iterators to transform the data before feeding it to the model.

* :ref:`The transformer pipeline<transformer_pipeline>`

The transformation pipeline is an extension of the sklearn pipelines for handling PM problems.
A pipeline is composed by a number of different transformation steps

* :ref:`The transformer step<transformer_step>`

Each step represent a transformation of the data.




