"""
En este programa se toma una serie temporal con precios de apertura/cierre y volumen de ventas del
petróleo. Se confeccionarán una regresión múltiple para obtener la predicción
de los valores. 
"""


#Cargamos las librerías a utilizar en nuestro modelo
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import sklearn
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor


#Leemos el dataset y realizamos un análisis exploratorio del mismo
petroleo=pd.read_csv("data/petroleo.csv",sep=',')
print("Primeras 20 líneas del dataset:\n{}".format(petroleo.head(20)))
print("Últimas 20 líneas del dataset:\n{}".format(petroleo.tail(20)))
print("Nombre de las variables:{}".format(petroleo.columns))
print("Descripción de las variables:{}".format(petroleo.describe))
print("Forma del dataset:{}".format(petroleo.shape))
petroleo=petroleo.dropna()
petroleo.to_excel("petro.xls")

#Algunos gráficos de interés
plt.plot(petroleo["Date"],petroleo["Volume"],'b-')
plt.xlabel("Fecha",size=10)
plt.ylabel("Volumen",size=10)
plt.title("Evolución del volumen de ventas del petróleo")
plt.xticks("off")
plt.show()

plt.hist(petroleo["Volume"],color="green",density=False,bins=40)
plt.title("Histograma de volúmenes de ventas",size=15)
plt.show()

sns.boxplot(petroleo["Volume"])
plt.title("Boxplot de volúmenes de ventas",size=15)
plt.show()

for i in [ 'Open', 'High', 'Low', 'Close', 'Adj Close']:
    plt.plot(petroleo["Date"],petroleo[i],label=i)
plt.xlabel("Fecha",size=10)
plt.ylabel("Price",size=10)
plt.title("Evolución del precio de ventas del petróleo",size=18)
plt.xticks("off")
plt.legend(loc='best')
plt.show()

para_graf=np.array([[0,0]])
for i in [ 'Open', 'High', 'Low', 'Close', 'Adj Close']:
    auxiliar=np.hstack((np.asmatrix(petroleo[i]).reshape(-1,1),np.asmatrix(np.full(petroleo.shape[0],i)).reshape(-1,1)))
    para_graf=np.concatenate((para_graf,auxiliar),axis=0)

print(para_graf)
petroleo.pop("Date")

#Pasamos ahora a una regresión múltiple sencilla

random.seed=0
Y=petroleo.pop("Volume")
X_train, X_test, Y_train, Y_test = train_test_split(petroleo,Y,random_state=0)
num_cols=['Open', 'High', 'Low', 'Close', 'Adj Close']
num_cols_2=["Volume"]
si_step=('si',SimpleImputer(strategy="mean"))
std_step=("std",StandardScaler())
num_steps=[si_step,std_step]
num_pipe=Pipeline(num_steps)
para_num_trans=[('num',num_pipe,num_cols)]
num_trans=ColumnTransformer(transformers=para_num_trans)
X_train_num=num_trans.fit_transform(X_train)
X_test_num=num_trans.transform(X_test)
print(X_train_num.shape)
print(X_test_num.shape)

A1=pd.DataFrame(X_train_num)
A2=pd.DataFrame(Y_train)
A1.to_excel("A1.xls")
A2.to_excel("A2.xls")

modelo_1=LinearRegression()
modelo_1.fit(X_train_num,Y_train)
print("MSE del modelo 1:{}".format(mean_squared_error(modelo_1.predict(X_test_num),Y_test)))
print("R2 del modelo 1:{}".format(r2_score(modelo_1.predict(X_test_num),Y_test)))
print("Ecuación del modelo:{}".format(modelo_1.get_params))

#Graficamos los puntos reales y los predichos

plt.plot(range(X_test_num.shape[0]),modelo_1.predict(X_test_num),'ro')
plt.plot(range(X_test_num.shape[0]),Y_test,'b*')
plt.title("Valor real vs predicción")
plt.show()
