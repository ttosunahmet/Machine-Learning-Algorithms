import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

data.drop(["id","Unnamed: 32"],
          axis=1,
          inplace=True)
#0print(data.tail())

M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

plt.scatter(M.radius_mean,M.texture_mean,color="red",label="kotu")
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="iyi")
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.legend()
#plt.show()
plt.close()

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)

x = (x_data - np.min(x_data)) / (np.max(x_data - np.min(x_data)))

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=1)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100 , random_state=1) # n_estimators = tree sayısı
rf.fit(x_train,y_train)

print("Accuracy for random algo :", rf.score(x_test,y_test))

y_pred = rf.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)


import seaborn as sns

f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
