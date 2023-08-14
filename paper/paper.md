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

`CeRULEo`, which stands for Comprehensive utilitiEs for Remaining Useful Life Estimation methOds, is a Python package designed to train and evaluate regression models for predicting remaining useful life (RUL) of equipment. RUL estimation is a process that uses prediction methods to forecast the future performance of machinery and obtain the time left before machinery loses its operation ability.  The RUL  estimation has been considered as a central 
technology of Predictive Maintenance (PdM) [@heimes2008recurrent; @li2018remaining].  PdM  techniques can statistically evaluate a piece of equipment's health status,  enabling early identification of impending failures and prompt pre-failure  interventions, thanks to prediction tools based on historical data [@susto2014machine].  `CeRULEo` offers a comprehensive suite of tools to help with the analysis and pre-processing of preventive maintenance data. These tools also enable the training and evaluation of RUL models that are tailored to the specific needs of the problem at hand. 

 
# Statement of need

Effective maintenance management helps reduce costs related to defective products and equipment downtime. A well-planned maintenance strategy improves reliability, prevents unexpected outages, and lowers operating costs. 


In Industry 5.0, the industrial machines produce a large amount of data which can be used to predict an asset’s life [@khan2023changes]. RUL estimation uses prediction techniques to forecast a machine's future performance based on historical data, enabling early identification of potential failures and prompt pre-failure interventions. 

Within the PdM and RUL regression ecosystem, finding a library that effectively combines modelling, feature extraction capabilities, and tools for model comparison poses a significant challenge. While numerous repositories and libraries exist for models and feature extraction in time series data [@christ2018time; @JMLR:v21:20-091], few offer a comprehensive solution that integrates both aspects effectively. The prog_models and prog_als libraries from NASA [@2022_nasa_prog_models] come closest to fulfilling this requirement. However, they have a strong focus on simulation and lack extensive mechanisms for feature extraction from time series data. 

On the other hand, `CeRULEo` provides a comprehensive set of utilities designed to train and evaluate regression models for predicting RUL of equipment. `CeRULEo`  emphasizes a data-driven approach using industrial data, particularly when a simulation model is unavailable or costly to develop, prioritizing model library-agnosticism for easy deployment in any production environment. 

In order to achieve good performance, RUL regression requires data preparation and feature engineering. Typically, machinery data is provided as time series data from various sensors during operation. The first step in data preparation is often to create a dataset based on run-to-failure cycles. This involves dividing the time series into segments where the equipment starts in a healthy state and ends in a failure state, or is close to failure. The second step of data preparation is preprocessing. While PdM models can be used in a variety of contexts with different data sources and errors, there are some general techniques that can be applied [@serradilla2022deep], such as time-series validation, imputing missing values, handling homogeneous or non-homogeneous sampling rates, addressing values, range and behaviour differences across different machines and the creation of run-to-failure-cycle-based data. 


`CeRULEo` addresses these issues by providing a comprehensive toolkit for preprocessing time series data for use in PdM models, with a focus on run-to-failure cycles. The preprocessing includes sensor data validation methods, for studying not only missing and corrupted values but also distribution drift among different pieces of equipment. 

In addition to preprocessing, it enables the iteration of machine data for use in both mini-batch and full-batch regression models, and is compatible with popular machine learning frameworks such as scikit-learn [@scikit-learn] and tensorflow [@tensorflow2015-whitepaper]. The library also includes a catalog of successful deep learning models [@jayasinghe2019temporal; @li2020remaining; @CHEN2022104969] from the literature and a collection of commonly used RUL datasets for quick model evaluation.

The acceptance of PdM technologies is pivotal in Industry 5.0 for successful implementation, but hesitations or reluctance by decision-makers  can still pose significant barriers [@van2022predictive]. One effective approach to foster acceptance and understanding is through explainability, which plays a crucial role in PdM.
As such, `CeRULEo`  incorporates explainable models capable of providing additional information about the predictions, enhancing comprehension: one that can select the most relevant features for the model [@lemhadri2021lassonet], and a convolutional model [@fauvel2021xcm] that provides post-hoc explanations of the predictions to understand the reasoning behind the predicted RUL. 

Moreover, `CeRULEo` provides tools for evaluating and comparing PdM models based on not only traditional regression metrics, but also on their ability to prevent errors and reduce costs. In many cases, the costs of not accurately detecting or anticipating faults can be much higher than the cost of inspections or maintenance due to reduced efficiency, unplanned downtime, and corrective maintenance expenses. In PdM, it is particularly important to be accurate about the RUL  of equipment near the end of its lifespan, as an overestimation of RUL can have serious consequences when immediate action is required. `CeRULEo` addresses this issue by providing mechanisms for weighting samples according to their importance and asymmetric losses for training models, as well as visualization tools for understanding model performance in relation to true RUL.



# Financial Acknowledgement

This work was partially carried out within the MICS (Made in Italy – Circular and Sustainable) Extended Partnership and received funding from Next-GenerationEU (Italian PNRR – M4 C2, Invest 1.3 – D.D. 1551.11-10-2022, PE00000004). Moreover this study was also partially carried out within the PNRR research activities of the consortium iNEST (Interconnected North-Est Innovation Ecosystem) funded by the European Union Next-GenerationEU (Piano Nazionale di Ripresa e Resilienza (PNRR) – Missione 4 Componente 2, Investimento 1.5 – D.D. 1058 23/06/2022, ECS00000043). This work was also co-funded by the European Union in the context of the Horizon Europe project 'AIMS5.0 - Artificial Intelligence in Manufacturing leading to Sustainability and Industry5.0' Grant agreement ID: 101112089.

# References
