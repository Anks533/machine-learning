import pandas as pd
import numpy as np
import streamlit as st
import math

## compute the cost of linear regression based on weight and bias.
def compute_cost(X,y,w,b):
    m = X.shape[0]
    total_cost = 0
    cost = 0
    for i in range(m):
        fwb_x = np.dot(w,X[i]) + b
        error = (fwb_x - y[i]) ** 2
        cost += error
    total_cost = (1/(2*m))*cost
    return total_cost
 
## compute gradient for given weight and bias.
def compute_gradient(X, y, w, b):
    m = X.shape[0]
    dj_db = 0
    dj_dw = 0
    for i in range(m):
        error = (np.dot(w,X[i]) + b) - y[i]
        dj_db = dj_db + error
        dj_dw = dj_dw + error * X[i]
    dj_db = (1/m) * dj_db
    dj_dw = (1/m) * dj_dw
    return dj_dw, dj_db

def gradient_descent(X, y, w, b, alpha, num_iters):
    J_history = []
    w_history = []
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X,y,w,b)
        w = w - alpha * dj_dw ## adjust
        b = b - alpha * dj_db ## adjust
        J_history.append(compute_cost(X,y,w,b))
        w_history.append(w)
        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
    return w, b, J_history, w_history

data = pd.read_csv("data/salary_data.csv")
X = data["YearsExperience"]
y = data["Salary"]

w,b,_,_ = gradient_descent(X,y,0,0,0.01,1500)
print("Slope: (w) ", w)
print("Intercept: (b) ", b)

m = X.shape[0]
y_hat = np.zeros(m)
for i in range(m):
    y_hat[i] = np.dot(w,X[i])+b

st.title("Salary Predictor")
years = st.slider("Years of Experience", 0, 20)
salary = np.dot(years,w) + b
st.write(f"Predicted Salary: ${salary}")
