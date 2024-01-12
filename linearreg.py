import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
df = pd.read_csv(r"/home/student/Desktop/ajnaas/lois_continuous.linear.csv")

c = 'Conductivity 25C continuous'
t = 'Temperature water continuous'
df = df.dropna(subset = [c,t])
x = df[[t]]
y = df[c]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.5, random_state = 1)

clf = LinearRegression()
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

    
mse = mean_squared_error(y_test,y_pred)

print("RMSE :\n",np.sqrt(mse))

plt.scatter(x, y, label="Actual")
plt.plot(x_test, y_pred, label="Predicted", color="red")
plt.xlabel("Temperature (°C)")
plt.ylabel("Conductivity 25C continuous")
plt.title("Linear Regression of Conductivity 25C continuous")
plt.show()


