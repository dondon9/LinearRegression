'''
this code is made to implemnt a linear reggretion model on Ecomerce data.
at first we will import the useful libraries to build the model
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
import seaborn as sb
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn import metrics

#star with reading the data 
customrs=pd.read_csv(r"/home/dndn/Desktop/Lin/Ecommerce Customers.csv") 
customrs.head()
customrs.describe()
customrs.info()

Y=customrs['Yearly Amount Spent']
X=customrs[['Avg. Session Length','Time on App','Time on Website', 'Length of Membership']]

#Now we are going to build the Linear regrission model
X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.3, random_state=100)
lr=LinearRegression()

#Train the model
lr.fit(X_train,Y_train)

#Test the model using predictions
predictions=lr.predict(X_test)

#Visualize the Linear Regression model
sb.set_palette("plasma")
sb.set_style("whitegrid")
mpl.scatter(Y_test,predictions)
mpl.xlabel('Y test')
mpl.ylabel('Predicted Y')
mpl.show()

#ACC metrics
print('MAE:', metrics.mean_absolute_error(Y_test, predictions))
print('MSE:', metrics.mean_squared_error(Y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))


