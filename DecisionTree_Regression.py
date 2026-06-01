import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("decision+tree+regression+dataset.csv", sep = ";", header=None)

x = df.iloc[:,0].values.reshape(-1,1)
y= df.iloc[:,1].values.reshape(-1,1)

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(x,y)

x_ = np.arange(np.min(x), np.max(x), 0.01).reshape(-1, 1)
y_head = tree_reg.predict(x_)

plt.scatter(x,
            y,
            color="red")
plt.plot(x_,
         y_head,
         color="green")
plt.xlabel("Tribun")
plt.ylabel("Ucret")
plt.show()