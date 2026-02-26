# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries and load the dataset using pandas.
2.Select input features (enginesize, horsepower, citympg, highwaympg) and target variable (price).
3.Split the dataset into training and testing sets (80% training, 20% testing).
4.Create a Linear Regression pipeline with StandardScaler, train the model, and predict test data.
5.Create a Polynomial Regression (degree 2) pipeline, train the model, and predict test data.
6.Evaluate both models using MSE, MAE, and R² score.
7.Plot actual vs predicted prices to compare Linear and Polynomial regression performance.  

## Program:
```
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: Varoodhini.M
RegisterNumber: 212225220118
*/
import pandas as pd
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt
df=pd.read_csv('encoded_car_data (1).csv')
print(df.head())
X=df[['enginesize','horsepower','citympg','highwaympg']]
Y=df['price']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
lr=Pipeline([
    ('scaler',StandardScaler()),
    ('model',LinearRegression())
])
lr.fit(X_train,Y_train)
y_pred_linear=lr.predict(X_test)
poly_model=Pipeline([
    ('poly',PolynomialFeatures(degree=2)),
    ('scaler',StandardScaler()),
    ('model',LinearRegression())
])
poly_model.fit(X_train,Y_train)
y_pred_poly=poly_model.predict(X_test)

print('Name: Varoodhini M')
print('Reg no: 212225220118')
print('Linear Regression:')
print('MSE=',mean_squared_error(Y_test,y_pred_linear))
print('MAE=',mean_absolute_error(Y_test,y_pred_linear))
r2score=r2_score(Y_test,y_pred_linear)
print('R2 Score=',r2score)

print("\nPolynomial Regression:")
print(f"MSE: {mean_squared_error(Y_test,y_pred_poly):.2f}")
print(f"R2:{r2_score(Y_test,y_pred_poly):.2f}")
plt.figure(figsize=(10,5))
plt.scatter(Y_test,y_pred_linear,label='Linear',alpha=0.6)
plt.scatter(Y_test,y_pred_poly,label='polynomial(degree=2)',alpha=0.6)
plt.plot([Y.min(),Y.max()],[Y.min(),Y.max()],'r--',label='Perfect Prediction')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()


```

## Output:
<img width="324" height="253" alt="image" src="https://github.com/user-attachments/assets/164947bf-d2f0-4b1e-9795-fe5502cca55b" />
<img width="1116" height="591" alt="image" src="https://github.com/user-attachments/assets/5d5d811d-92df-4a52-ad5a-0c5442f54baa" />




## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
