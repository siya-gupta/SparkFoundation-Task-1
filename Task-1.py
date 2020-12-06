# SparkFoundation-Task-1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
%matplotlib inline
url="http://bit.ly/w-data"
dataset = pd.read_csv(url)
dataset.describe()
X = dataset.iloc[ : ,   : 1 ].values
Y = dataset.iloc[ : , 1 ].values
plt.scatter(dataset.Hours, dataset.Scores)
plt.xlabel('No. of Hours')
plt.ylabel('Scores')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, train_size=0.80,test_size=0.20,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color= 'blue')
plt.title('Hours vs Scores(Training set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color= 'blue')
plt.title('Hours vs Scores(Test set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
print('Score of student who studied for 9.25 hours a dat', regressor.predict([[9.25]]))
r2_score(y_test, y_pred)
from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))
