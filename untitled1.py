# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd




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
