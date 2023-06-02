---
title: 'CeRULEo: Comprehensive utilitiEs for Remaining Useful Life Estimation methOds'
tags:
  - Python
  - predictive maintenance
  - remaining useful life  
authors:
  - name: Luciano Lorenti
    orcid: 0000-0002-4041-449X
    affiliation: 1
    corresponding: true
  - name: Gian Antonio Susto
    orcid: 0000-0001-5739-9639
    affiliation: 1
affiliations:
  - name:
      Department of Engineering, University of Padova.
      Italy
    index: 1
date: 20 Dic 2022
bibliography: paper.bib
---

# Summary

`CeRULEo`, which stands for Comprehensive utilitiEs for Remaining Useful Life Estimation methOds, is a Python package designed to train and evaluate regression models for predicting remaining useful life (RUL) of equipment. RUL estimation is a process that uses prediction methods to forecast the future performance of machinery and obtain the time left before machinery loses its operation ability.  The remaining useful life  estimation has been considered as a central 
technology of Predictive Maintenance (PdM) [@heimes2008recurrent; @li2018remaining].  PdM  techniques can statistically evaluate a piece of equipment's health status,  enabling early identification of impending failures and prompt pre-failure  interventions, thanks to prediction tools based on historical data [@susto2014machine].  `CeRULEo` offers a comprehensive suite of tools to help with the analysis and pre-processing of preventive maintenance data. These tools also enable the training and evaluation of RUL models that are tailored to the specific needs of the problem at hand. 

 
# Statement of need

Effective maintenance management helps reduce costs related to defective products and equipment downtime. A well-planned maintenance strategy improves reliability, prevents unexpected outages, and lowers operating costs. In Industry 4.0, data from the manufacturing process can enhance decision-making. RUL estimation uses prediction techniques to forecast a machine's future performance based on historical data and determine its remaining useful life, enabling early identification of potential failures and prompt pre-failure interventions. In this context, `CeRULEo` provides a comprehensive set of utilities designed to train and evaluate regression models for predicting remaining useful life of equipment. 

In order to achieve good performance, RUL regression requires data preparation and feature engineering. Typically, machinery data is provided as time series data from various sensors during operation. The first step in data preparation is often to create a dataset based on run-to-failure cycles. This involves dividing the time series into segments where the equipment starts in a healthy state and ends in a failure state, or is close to failure. The second step of data preparation is preprocessing. While PdM models can be used in a variety of contexts with different data sources and errors, there are some general techniques that can be applied [@serradilla2022deep], such as time-series validation, imputing missing values, handling homogeneous or non-homogeneous sampling rates, addressing values, range and behaviour differences across difference machines and the creation of run-to-failure-cycle-based data. 


`CeRULEo` addresses these issues by providing a comprehensive toolkit for preprocessing time series data for use in PdM models, with a focus on run-to-failure cycles. The preprocessing includes sensor data validation methods, for studying not only missing and corrupted values but also distribution drift among different pieces of equipment. 

In addition to preprocessing, it enables the iteration of machine data for use in both mini-batch and full-batch regression models, and is compatible with popular machine learning frameworks such as scikit-learn [@scikit-learn] and tensorflow [@tensorflow2015-whitepaper]. The library also includes a catalog of successful deep learning models [@jayasinghe2019temporal; @li2020remaining; @CHEN2022104969] from the literature and a collection of commonly used remaining useful life datasets for quick model evaluation.

In the context of predictive maintenance, explainability is crucial. As such, `CeRULEo` includes two explainable models: one that can select the most relevant features for the model [@lemhadri2021lassonet], and a convolutional model [@fauvel2021xcm] that provides post-hoc explanations of the predictions to understand the reasoning behind the predicted remaining useful life. This helps users better understand and trust the model's predictions.

Moreover, `CeRULEo` provides tools for evaluating and comparing PdM models based on not only traditional regression metrics, but also on their ability to prevent errors and reduce costs. In many cases, the costs of not accurately detecting or anticipating faults can be much higher than the cost of inspections or maintenance due to reduced efficiency, unplanned downtime, and corrective maintenance expenses. In predictive maintenance, it is particularly important to be accurate about the remaining useful life  of equipment near the end of its lifespan, as an overestimation of RUL can have serious consequences when immediate action is required. `CeRULEo` addresses this issue by providing mechanisms for weighting samples according to their importance and asymmetric losses for training models, as well as visualization tools for understanding model performance in relation to true RUL.


# Financial Acknowledgement

The Italian Government PNRR iniatiatives 'Partenariato 11: Made in Italy circolare e sostenibile' and 'Ecosistema dell'Innovazione - iNest' are gratefully acknowledged for partially financing this research activity.

# References
