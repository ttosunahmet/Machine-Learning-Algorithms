import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("linear_regression_dataset.csv", sep = ";")

plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
# plt.show()
plt.close()

from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

b0 = linear_reg.predict([[0]])
print("b0 :", b0)

b0_ = linear_reg.intercept_
print("b0_ :", b0_)

b1 = linear_reg.coef_
print("b1 :", b1)

maas_yeni = b0 + b1*11
print(maas_yeni)

array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)

plt.scatter(x,y)

y_head = linear_reg.predict(array)

y_pred = linear_reg.predict(x)

plt.plot(array,
         y_head,
         color = "red")
#plt.show()
plt.close()

from sklearn.metrics import r2_score
print("r_score : ", r2_score(y,y_pred))