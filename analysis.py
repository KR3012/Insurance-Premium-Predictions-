# Insurance Premium Prediction Model
# using Multiple Linear Regression
# Predicts a customer's estimated insurance premium 
# based on Age, Years Driving, Previous Claims and Annual Mileage.
# By plotting data on a graph model finds gradient and constant finding a linear regression line 
# subbing back into equation you can find an estimated insurance premium for any custormer of any age

import pandas as pd # pandas toolbox (pd is short for pandas)
import matplotlib.pyplot as plt # plt short name for matplotlib 
# matplotlib is the drawing tool 
from sklearn.linear_model import LinearRegression
# sklearn.linear_model part original scikit-learn that deals with models
# LinearRegression only using linearRegression in that toolbx
data = pd.read_excel("InsurancePricingData_v1.xlsx")
pd.set_option("display.max_columns",None)
print(data.head())
print(data["FinalPremium"].mean()) #calculating the means for all Final Premiums
X = data[["Age", "Years Driving", "Previous Claims", "Annual Mileage"]]
Y = data["FinalPremium"]
model = LinearRegression()
model.fit(X, Y)
predicted_premium = model.predict(X)

plt.scatter(data["Age"], data["FinalPremium"], label="Actual Premiums") 
# scatter means plotting dots as customers 
# plt.plot(data["Age"], predicted_premium)
plt.scatter(data["Age"], predicted_premium, label="Predicted Premiums")

plt.xlabel("Age")
plt.ylabel("Final Premium (£)")
plt.title("Age/Final Premium")
plt.legend() # plots key of what colour is what
# gradient (m) of line is Δy/Δx therefore finalpremiums/age 
# Equation y=mx+c is now Estimated Premiums = m x Age + c
print("Gradient:", model.coef_[0])
print("Intercept:", model.intercept_)
print(f"Equation of line: Estimated Premium = {model.coef_[0]:.2f} x Age + {model.intercept_:.2f}")
plt.savefig("Insurance_Premium_Graph.png")
plt.show()


