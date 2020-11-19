# COV-(AI)D

Coronaviruses are a group of viruses that can cause illness in animals and humans. Coronavirus Disease 2019 (COVID-19) is a disease that was identified in Wuhan, China and is now being spread throughout the world. In this project, I tried to develop a model using deep-learning methods to predict COVID-19 case growth in U.S. states.

By utilizing deep learning techniques, we study past COVID-19 case trends to forecast case growth in US States with a prediction window of 30 days. In addition to past COVID data, the model uses other data points that may influence disease transmission, such as human mobility trends in each state. Compared to traditional models for time-series analysis like ARIMA, deep learning methods as employed in this model are often better at conforming to complex nonlinear dependencies within the data. It was also more applicable in this project as I was analyzing time-series data from two sources, not just past case trends.

# Data Collection

The key data points collected were COVID-19 case counts and Human Mobility data. Upon viewing the mobility trends and COVID-19 case counts for different states, it seemed as if high rates of human mobility (lack of social distancing) influenced spikes in COVID-19 cases. The sources of the datasets are described below:

## COVID-19 Case Counts
COVID-19 case count data was collected from the  Johns Hopkins University Center for Systems Science and Engineering (CSSE), as they have done a wonderful job of tracking COVID cases around the world. The case counts were cleaned and normalized before being fed to the model, but you can view & download the data here:
https://datahub.io/core/covid-19

## Human Mobility Data
As increased rates of COVID-19 transmission may be associated with a lack of social distancing, human mobility data was collected to provide more insight to the movement of people. Apple has collected mobility trend data all around the world in the past several months by tracking routing requests to Maps. The percentage increase/decrease in routing requests each day was fed to the model in conjunction with COVID-19 case trends. The data can be accessed here:
https://www.apple.com/covid19/mobility

Note: Google also has a similar data set where they track multiple metrics including retail & recreation, residential, workplaces, grocery & pharmacy, and transit stations. Access the Google mobility reports here:
https://www.google.com/covid19/mobility/

# Model Summary

A popular deep learning approach to recognize sequential characteristics within data is to use a Recurrent Neural Network (RNN)- in this case, it has been applied to multivariate time-series data. Unlike a classical neural network or MLP (Multilayer Perceptron), RNNs were designed to encode sequential information using a recurrent unit. Essentially, the model maintains a vector which learns to describe key information from past time-steps. For instance, the model could use COVID-19 case counts and mobility data from the past 30 days to predict total cases for the next day. Although RNNs are mostly used in Natural Language Processing (like in your iPhone keyboard's word prediction feature), they can be applied to time-series data as well. This model uses a 3-layer LSTM (a type of RNN) to process and forecast COVID-19 trends.

The objective function used was SmoothL1 loss, which is differentiable at zero unlike MAE (L1 Loss) and is not as sensitive to outliers as RMSE. The optimizer used was Adam. Sequences of length 30 were fed to the model from different US states, as this sequence length seemed to perform best. Dropout of 0.1 was applied on the final LSTM layer.

Overall, the model performed considerably well, with an MAE of around 600. The final predicted trendlines were fit with a smoothing spline to reduce noise in the data. However, there are still some inconsistencies in the predictions at later dates - as this model was an experiment, any suggestions for improvement (see contact page) would be appreciated.

# Project Site

The project website is linked here: https://covidforecast.weebly.com/

Unfortunately, I don't have a real domain at the moment, but this will hopefully be changed soon. (Note: the site was active from May 2020, but if you are viewing this post-2020 or post-COVID the site may be outdated/removed).

TODO: Build automated web driver to scrape data, run model and update site
