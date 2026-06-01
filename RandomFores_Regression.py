import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("random+forest+regression+dataset.csv", sep = ";", header = None)

from sklearn.ensemble import RandomForestRegressor

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x, y.ravel())

y_pred = rf.predict(x)


x_ = np.arange(np.min(x), np.max(x), 0.01).reshape(-1,1)
y_head = rf.predict(x_)

plt.scatter(x,
            y,
            color="red")
plt.plot(x_,
         y_head,
         color="green")
plt.xlabel("Tribun")
plt.ylabel("Fiyat")
# plt.show()
plt.close()

from sklearn.metrics import r2_score
print("r_score :" , r2_score(y,y_pred))