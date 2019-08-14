import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
#import seaborn as sns
import os
import sys

from get_data import *
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas.plotting import autocorrelation_plot



#plt.plot(ps_open_t_arr[0])
#plt.show()
#print("Actual series plotted")
#autocorrelation_plot(ps_open_t_arr[0])
#plt.show()
#print("Autocorrelation also plotted")



TRAIN_TEST_RATIO = 0.66
size = int(len(ps_open_t_arr[0]) * TRAIN_TEST_RATIO)
train, test = ps_open_t_arr[0][0:size], ps_open_t_arr[0][size:len(ps_open_t_arr[0])]
history = [x for x in train]
predictions = list()
error_list = list()
for t in range(len(test)):
	model = ARIMA(history, order=(1,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	#yhat = output[0]
	predictions.append(output[0])
	#obs = test[t]
	history.append(test[t])
	error_list.append(mean_squared_error(test[0:(t+1)], predictions))
	print("Running prediction for t=",t)
    #print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
plt.plot(error_list)
plt.show()
plt.plot(test, color='blue')
plt.plot(predictions, color='red')
plt.show()







