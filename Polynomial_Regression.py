import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("polynomial+regression.csv", sep = ";")

x = df.araba_fiyat.values.reshape(-1,1)
y = df.araba_max_hiz.values.reshape(-1,1)

plt.scatter(x,y)
plt.ylabel("araba_max_hiz")
plt.xlabel("araba_fiyat")
#plt.show()

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)

y_head = lr.predict(x)

plt.plot(x,
         y_head,
         color="red",
         label="linear")
#plt.show()

from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = PolynomialFeatures(degree=4)

x_polynomial = polynomial_regression.fit_transform(x)

linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial,y)

y_head2 = linear_regression2.predict(x_polynomial)

plt.plot(x,
         y_head2,
         color="green",
         label="poly")
plt.legend()
plt.show()