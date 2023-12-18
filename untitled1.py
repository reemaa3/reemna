# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd


data=pd.read_csv('economic_data.csv')
print(data)

print(data.describe())

print(data.head())
plt.scatter(data['Year'],data['GDP'])
plt.show()

#y=mx+b

print(data.head())

x=data.iloc[:,:1]  
y=data.iloc[:,1]

print(x)
print(y)

from sklearn.linear_model import  LinearRegression
model =LinearRegression()
model.fit(x,y)
#m
print(model.coef_)
#b
print(model.intercept_)

#GPD(y)=model.coef_ * Year(x) + model.intercept_
plt.scatter(x,y)
plt.plot(x,model.predict(x),'r')



model.predict([[2]])


#valdation
model.score(x,y)