#%% İmport library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Kütüphaneleri Projemize Aktardık.
# We Transferred the Libraries to Our Project.


df = pd.read_csv("data.csv")
df.drop(["Unnamed: 32","id"],
        axis=1,
        inplace=True)
# Unnamed: 32 ve id isimli kolonları kaldırdık. axis=1 tüm sütunların alınacağını ve inplace=True ifadesi ise bu ayarların kaydedilmesi gerektiğini söylemektedir.
# Unnamed: We removed the columns named 32 and id. axis=1 means that all columns will be imported, and inplace=True means that these settings should be saved.

df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]
# diagnosis kolonundaki M ve B ifadelerini integer değeler olarak atıyoruz.
# We assign the M and B expressions in the diagnosis column as integer values.
#print(df.info())

y=df.diagnosis.values
x_data = df.drop(["diagnosis"],axis=1)

x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))
# Normalize ederek değerler arasında oluşan yüksek farkları en düşük derecelere indirgeyip sonucumuzun daha kolay çıkmasını sağlıyoruz.
# By normalizing, we reduce the high differences between values ​​to the lowest degrees and make our results easier to obtain.
print(x)

#%% train test split
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# Oluşturduğumuz modellerimizin doğruluğunu hesaplayabilmemiz için x_test ve y_test adında değişkenler oluşturuyoruz ve bunları %20 oranında kendi değerlerimizi ise %80 oranında olacak şekilde ayarlıyoruz.random_state ile aynı kod tekrar çalıştırıldığında aynı bölme yapılır.
# In order to calculate the accuracy of the models we have created, we create variables named x_test and y_test and set them to 20% and our own values ​​to 80%. When the same code is run again with random_state, the same division is made.

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T
# .T ile değer kümelerinin x ve y değerlerinin yerlerini değiştiriyoruz.
# With .T we swap the x and y values ​​of the value sets.

print("x_train shape : ", x_train.shape)
print("x_test shape : ", x_test.shape)
print("y_train shape : ", y_train.shape)
print("y_test shape : ", y_test.shape)


#%% parameter initialize and sigmoid function

def initialize_weights_and_bias(dimension):
        
        w = np.full((dimension,1),0.01)
        b = 0.0
        return w,b
# np.full(shape, fill_value): NumPy'da, verilen boyutta bir matris oluşturur ve tüm elemanları fill_value ile doldurur.

def sigmoid(z):
        
        y_head = 1/(1+np.exp(-z))
        # exp = e 
        # sigmoid fonksiyonu oluşturuyoruz.
        
        return y_head

# %%
def forward_bacward_propagation(w,b,x_train,y_train):
        # Forward Propagation
        z = np.dot(w.T,x_train) + b # Sum işlemini gerçekleştiriyoruz w ve x_train değerlerimizi .dot ile çarpıyoruz sonrasında bias  değeri ile topluyoruz bunun ardından sigmoid fonksiyonu içine atıcaz.
        y_head = sigmoid(z)
        loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head) # Loss(kayıp) denklemini yazıyoruz.(internette yazıyor kendimiz oluşturduğumuz bir fonksiyon değil.)
        cost = (np.sum(loss))/x_train.shape[1] # toplam kayıp / örnek sayısı bölümü sonucunda maliyet(cost) bulunur.Burada, x_train.shape[1], genelde eğitim örneklerinin toplam sayısını ifade eder.
        # shape[0]: Satır sayısı (örnek sayısı).
        # shape[1]: Sütun sayısı (özellik sayısı).
        
        # Bacward Propagation
        derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
        derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
        gradients = {"derivative_weight": derivative_weight , "derivative_bias": derivative_bias}
        
        return cost,gradients

#%% Updating(learning) parameters
def update(w, b, x_train, y_train, learning_rate, number_of_iterarion):
        # Öğrenme derecesi ve İşlemlerin forward_backward_propagation kaç kere yapılacağını number_of_iterarion ile belirtiriz.
        cost_list = []
        cost_list2 = []
        index = []
        
        # updating[learning] parameters is number_of_iterarion times
        for i in range(number_of_iterarion):
                
                cost,gradients = forward_bacward_propagation(w,b,x_train,y_train)
                cost_list.append(cost)
                
                w= w- learning_rate * gradients["derivative_weight"]
                b= b- learning_rate * gradients["derivative_bias"]
                # Formül kullanıyoruz.
                if i % 10 == 0:
                        cost_list2.append(cost)
                        index.append(i)
                        print ("Cost after iteration %i: %f" %(i, cost))
                # 10 hatada 1 listeye kayıt ediyoruz.
                        
        parameters = {"weight": w,"bias": b}
        plt.plot(index,cost_list2)
        plt.xticks(index,rotation="vertical")
        plt.xlabel("Number of Iterarion")
        plt.ylabel("Cost")
        plt.show()
        return parameters, gradients, cost_list

#%% prediction

def predict(w,b,x_test):
        
        z = sigmoid(np.dot(w.T,x_test)+b)
        Y_prediction = np.zeros((1,x_test.shape[1]))
        
        for i in range(z.shape[1]):
                if z[0,i]<= 0.5:
                        Y_prediction[0,i] = 0
                        
                else:
                        Y_prediction[0,i] = 1
                        
        return Y_prediction

#%% Logistic Regression

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterarion):

        
        dimension = x_train.shape[0]
        w,b = initialize_weights_and_bias(dimension)
        
        parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterarion)
        
        y_prediction_test = predict(parameters["weight"], parameters["bias"],x_test)
        
        print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

logistic_regression(x_train, y_train, x_test, y_test, learning_rate=3, num_iterarion=300)

#%% sklearn with LR
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
print("test accuracy {} %".format(lr.score(x_test.T,y_test.T)))