# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
'''
Program to implement the simple linear regression model for predicting the marks scored.
Developed by : PULI NAGA NEERAJ
RegisterNumber : 212223240130


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
print("HEAD:")
print(df.head())
print("TAIL:")
print(df.tail())
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
import matplotlib.pyplot as plt
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='yellow')
plt.title("Hours Vs Scores(Train Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="green")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.title("Hours vs scores (test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
MSE = mean_squared_error(Y_test,Y_pred)
print('MSE = ',MSE)
MAE = mean_absolute_error(Y_test,Y_pred)
print('MAE = ',MAE)
RMSE=np.sqrt(MSE)
print("RMSE = ",RMSE)
```
## Output:

![image](https://github.com/PuliNagaNeeraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849173/60f7555b-a3f1-44d5-89f0-0f7e22d58c18)

![image](https://github.com/PuliNagaNeeraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849173/14aeb90c-bd3a-42e9-b0de-50dd8b91e38d)

![image](https://github.com/PuliNagaNeeraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849173/b9881dda-4f52-437b-95c4-9b2a73181382)

![image](https://github.com/PuliNagaNeeraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849173/dce4427d-8043-4b70-9f87-59bcce58c269)

![image](https://github.com/PuliNagaNeeraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849173/38583901-2737-48f2-9b4c-0f7eabfbdcb6)

![image](https://github.com/PuliNagaNeeraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849173/57c1230e-6539-45f3-9a95-12c103adb7c6)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
