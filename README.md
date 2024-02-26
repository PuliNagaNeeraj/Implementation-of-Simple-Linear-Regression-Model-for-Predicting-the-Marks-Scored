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
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PULI NAGA NEERAJ
RegisterNumber: 212223240130
*/
import pandas as pd
df=pd.read_csv('student_scores.csv')
print(df.head())
print(df.tail())
x=(df.iloc[:,:-1]).values
x
y=(df.iloc[:,1]).values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test
import matplotlib.pyplot as plt
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours Vs Scores(Train Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="purple")
plt.plot(x_test,regressor.predict(x_test),color="yellow")
plt.title("Hours vs scores (test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

## Output:

![ML2 1](https://github.com/PuliNagaNeeraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849173/9e8a4983-31a0-415c-aff7-e1f09431db36)

![ML2 2](https://github.com/PuliNagaNeeraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849173/145455fd-55d4-4b2c-8005-09472c9061eb)

![ML2 3](https://github.com/PuliNagaNeeraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849173/f2475e7f-2512-45c3-942e-5f921a3194dc)

![ML2 3 1](https://github.com/PuliNagaNeeraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849173/9dd569d5-0592-496d-9d53-643c6632ea04)

![ML2 3 2](https://github.com/PuliNagaNeeraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849173/42cf2918-f3e4-4b16-801e-8a5912118e76)

![ML2 3 3](https://github.com/PuliNagaNeeraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849173/36a86afe-294b-42fe-b970-d40ad308679c)

![ML2 5](https://github.com/PuliNagaNeeraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849173/6fc68eaf-64d8-4252-a429-7de275628a20)

![ML2 6](https://github.com/PuliNagaNeeraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849173/d26ecd8b-f49b-4639-b8e6-620a2a20aa08)

![ML2 7](https://github.com/PuliNagaNeeraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849173/0aaa5593-c854-4bea-a163-9ac7616bd148)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
