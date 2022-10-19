# Master-Thesis: Probabilistic Day Ahead Forecasting of the German CO2 Emission Factor 

Nowadays, lowering $CO_{2}$ emissions is paramount to the common goal of combating climate change and working towards a sustainable future. Generating accurate short-term $CO_{2}$ forecasts can help achieve this objective by enabling the scheduling of storage and flexible electricity consumers of the power grid to minimize $CO_2$ emissions. Recent work used statistical approaches like SARMA and deep learning to predict the average $CO_{2}$ emission factor of the power grids of various European countries. This thesis aims to provide probabilistic, short-term forecasts of the $CO_2$ emission factor of the German power grid. Different statistical and machine learning models, such as SARMAX, ensemble learners, and neural networks, are created for forecasting the $CO_2$ emission factor time series. The prediction models are selected based on a created model selection methodology. The forecast models use market data forecasts, scheduled imports and exports, and lagged values of the $CO_2$ emission factor for their predictions. Afterward, the advantages and disadvantages of the built forecasting models are evaluated. Here, both predictive accuracy and uncertainty of the various models are considered for the evaluation process of the $CO_2$ emission factor prediction models. Finally, a heterogeneous ensemble combines statistical approaches with machine learning models.

The following python packages have been used to create this thesis:

* NumPy
* Pandas
* Matplotlib
* Statsmodels
* Scikit-learn
* Keras
* Tensorflow
* Visualkeras


