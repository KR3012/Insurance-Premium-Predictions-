import pandas as pd # pandas toolbox (pd is short for pandas)
import matplotlib.pyplot as plt # plt short name for matplotlib 
# matplotlib is the drawing tool 
from sklearn.linear_model import LinearRegression
# sklearn.linear_model part original scikit-learn that deals with models
# LinearRegression only using linearRegression in that toolbx
from sklearn.model_selection import train_test_split 
# train_test_split is a function that splits data into a teaching and testing pile
# this allows us to test the model on data it has never seen before
from sklearn.metrics import r2_score, mean_squared_error
# r2_score and mean_squared_error are functions that calculate how good the model is
# r2_score = R² tells us the variation of the data thereforehow far our prediction is from the average 
# having 0.616/ 61.6% tells us 61.6% of the variation in the data is explained by our model, the rest the model cannot explain for ex location 
# root_mean_squared_error = RMSE (on average how far off in £ our guess is form average)

data = pd.read_excel("InsurancePricingData_v1.xlsx")
pd.set_option("display.max_columns",None)
print(data.head())
print(data["FinalPremium"].mean()) #calculating the means for all Final Premiums
X = data[["Age", "Years Driving", "Previous Claims", "Annual Mileage"]]
Y = data["FinalPremium"]

# spliting customers into 80% to train and 20% to test on
# random_state=40 just makes the split the same every time we run it, this allows our results to be repeated 
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=40
)
model = LinearRegression()
model.fit(X_train, Y_train)
predicted_premium = model.predict(X_test)
# making the model guess premiums only for the 20% test, therefore the ones it has never seen before

r2 = r2_score(Y_test, predicted_premium)
rmse = mean_squared_error(Y_test, predicted_premium) ** 0.5
print(f"R²: {r2:.3f}")
print(f"RMSE: £{rmse:.2f}")

plt.scatter(X_test["Age"], Y_test, label="Actual Premiums") 
# scatter means plotting dots as customers 
# plt.plot(data["Age"], predicted_premium)
plt.scatter(X_test["Age"], predicted_premium, label="Predicted Premiums")
plt.xlabel("Age")
plt.ylabel("Final Premium (£)")
plt.title("Age/Final Premium")
plt.legend() # plots key of what colour is what
# gradient (m) of line is Δy/Δx therefore finalpremiums/age 
# Equation y=mx+c is now Estimated Premiums = m x Age + c
print("Gradient:", model.coef_[0])
print("Intercept:", model.intercept_)
print(f"Equation of line: Estimated Premium = {model.coef_[0]:.2f} x Age + {model.intercept_:.2f}")
plt.savefig("Insurance_Premium_Graph.png") # Saves an image of the graph plotted
print()


