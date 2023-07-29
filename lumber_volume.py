import pandas as pd
import matplotlib.pyplot as plt


from sklearn import linear_model
import numpy as np   
# read our data in using 'pd.read_csv('file')'

data_path  = 'dataset-14921.csv'
tree = pd.read_csv(data_path)



X = tree[['Girth', 'Height']]
y = tree['Volume']

linear = linear_model.LinearRegression(fit_intercept = True, normalize = True)



# train the model 
linear.fit(X, y)
print('Our multiple linear model had an R^2 of:', linear.score(X, y))
#strings
inp1 = input("Enter tree girth")
inp2 = input("Enter tree height, taken using standard measurement practices")

y_pred = linear.predict([[inp1, inp2]])
print('The predicted volume is:',y_pred)

