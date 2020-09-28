
""" Este programa toma un dataset con variables numércias y categóricas y le realiza las transformaciones
necesarias para obtener una dense matrix. Una vez llegado a este punto y sobre esa matriz se aplicarán
algoritmos de machine learning en una regresión que realice predicciones acerca del rating de diferentes
cereales. Se aplicarán modelos de regresión múltiple, árboles de decisión y random forest. La forma de
evaluar y comparar los diferentes modelos y entender cual elegir como el mejor será a través de las métricas
de R-Cuadrado y Mean squared Error (MSE). """


# encoding: utf-8
# Cargamos las librerias a utilizar

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV 


#Cargamos el dataset a utilizar y obtenemos algunos parámetros

cereales=pd.read_excel("https://raw.githubusercontent.com/fferrant/excel_files/master/cereales.xls",sep=";")
print("Primeras 10 líneas del dataset:\n {}".format(cereales.head(10)))
print("Últimas 10 líneas del dataset:\n{}".format(cereales.tail(10)))
print("Variables:\n{}".format(cereales.columns))
print("Forma del dataset: {}".format(cereales.shape))
print("Descripción del dataset: {}".format(cereales.describe()))
print("Cantidad de productos por empresa:\n{}".format(cereales.groupby("manufacturer")["manufacturer"].count()))
print("Cantidad de productos por tipo:\n{}".format(cereales.groupby("type")["type"].count()))


#Realizamos algunos gráficos descriptivos

sns.boxplot(y=cereales["rating_of_cereal"],x=cereales["manufacturer"])
plt.xlabel("Empresa productora",size=10)
plt.ylabel("Ranking",size=10)
plt.title("Boxplot de rating de cereales por empresa productora",size=20)
plt.xticks(rotation=90)
plt.show()

plt.scatter(cereales["grams_of_sugars"][cereales["type"]=="Cold"],cereales["rating_of_cereal"][cereales["type"]=="Cold"],label="cold",color="blue")
plt.scatter(cereales["grams_of_sugars"][cereales["type"]=="Hot"],cereales["rating_of_cereal"][cereales["type"]=="Hot"],label="hot",color="red")
plt.legend(loc="best")
plt.xlabel("Grs de azúcar",size=10)
plt.ylabel("Rating del cereal",size=10)
plt.title("Calidad del cereal en función del contenido de azúcar",size=20)
plt.show()


#Preparamos nuestros datos antes de aplicar los algoritmos de machine learning

Y=cereales["rating_of_cereal"]
X=cereales.iloc[:,1:15]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=0)

num_cols=['calories_per_serving',
       'grams_of_protein', 'grams_of_fat', 'milligrams_of_sodium',
       'grams_of_dietary_fiber', 'grams_of_complex_carbohydrates',
       'grams_of_sugars', 'milligrams_of_potassium']
cat_cols=['manufacturer', 'type']
bin_cols=['vitamins_and_minerals_of_fda_recommendation','display_shelf',
    'weight_in_ounces_per_one_serving','number_of_cups_in_one_serving']
 

"""Aplicamos ahora los pasos para transformar nuestro dataset en una dense matrix. Se realizarán 3
sub-transformaciones: una para las variables categóricas, otra para las variables numéricas continuas 
y la última para variables numéricas que se comportan mejor dentro de bins""" 


#Sub-transformación de las variables categóricas
si_step = ('si', SimpleImputer(strategy='constant', fill_value='MISSING'))
ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))
cat_steps=[si_step,ohe_step]
cat_pipe=Pipeline(cat_steps)
para_cat_trans = [('cat', cat_pipe, cat_cols)]
cat_trans=ColumnTransformer(transformers=para_cat_trans)
X_train_cat=cat_trans.fit_transform(X_train)
X_test_cat=cat_trans.transform(X_test)
print("\n\n\n Forma del train set categorico:{}".format(X_train_cat.shape))
print("Forma del test set categorico:{}".format(X_test_cat.shape))

#Sub-transformación de las variables numéricas continuas
si2_step=('si2',SimpleImputer(strategy="mean"))
std_step=("std",StandardScaler())
num_steps=[si2_step,std_step]
num_pipe=Pipeline(num_steps)
para_num_trans=[('num',num_pipe,num_cols)]
num_trans=ColumnTransformer(transformers=para_num_trans)
X_train_num=num_trans.fit_transform(X_train)
X_test_num=num_trans.transform(X_test)
print("Forma del train set numérico:{}".format(X_train_num.shape))
print("Forma del test set numérico:{}".format(X_test_num.shape))

#Sub-transformación de las variables numéricas a clusterizar
si3_step=('si3',SimpleImputer(strategy='median'))
kb_step=('kb',KBinsDiscretizer(encode='onehot-dense'))
bin_steps=[si3_step,kb_step]
bin_pipe=Pipeline(bin_steps)
para_bin_trans=[('bin',bin_pipe,bin_cols)]
bin_trans=ColumnTransformer(transformers=para_bin_trans)
X_train_bin=bin_trans.fit_transform(X_train)
X_test_bin=bin_trans.transform(X_test)
print("Forma del train set clusterizado:{}".format(X_train_bin.shape))
print("Forma del test set clusterizado:{}".format(X_test_bin.shape))


#Ahora realizamos la tranformación conjunta
para_trans_total=[('cat', cat_pipe, cat_cols),('num',num_pipe,num_cols),('bin',bin_pipe,bin_cols)]
trans_total=ColumnTransformer(transformers=para_trans_total)
X_train_final=trans_total.fit_transform(X_train)
X_test_final=trans_total.transform(X_test)
print("\n\n\n")
print("Forma del train set tranformado:{}".format(X_train_final.shape))
print("Forma del test set transformado:{}".format(X_test_final.shape))


"""Ahora que tenemos nuestro train set y test set transformados, podemos comenzar a aplicar
nuestros algoritmos de ML"""


# 1) Regresión múltiple
regresion=LinearRegression()
regresion.fit(X_train_final,Y_train)
result=regresion.predict(X_test_final)
print("RMSE del modelo de regresión: {}".format(mean_squared_error(Y_test,result)))
print("R2 del modelo de regresión: {}".format(r2_score(Y_test,result)))
print("\n\n")

# 2) Árbol de decisión
X_final=trans_total.fit_transform(X)
parametros_arbol = {
    "max_depth": range(1, 11, 2),
    "min_samples_split": range(2, 20, 2),
    "min_samples_leaf":range(1,11,2)
}
arbol=DecisionTreeRegressor(random_state=0)
modelo_arbol=GridSearchCV(arbol,parametros_arbol,cv=2)
modelo_arbol.fit(X_train_final,Y_train)
result_arbol=modelo_arbol.predict(X_test_final)
print("RMSE del árbol de regresión: {}".format(mean_squared_error(Y_test,result_arbol)))
print("R2 del árbol de regresión: {}".format(r2_score(Y_test,result_arbol)))
print("\n\n")

# 3) Random Forest
parametros_bosque = {
    "max_depth": range(1, 7, 2),
    "min_samples_split": range(2, 16, 2),
    "min_samples_leaf":range(1,9,2),
    "n_estimators":range(10,30,10)
}
bosque=RandomForestRegressor(random_state=0)
modelo_bosque=GridSearchCV(bosque,parametros_bosque,cv=2)
modelo_bosque.fit(X_train_final,Y_train)
result_bosque=modelo_bosque.predict(X_test_final)
print("RMSE del bosque de regresión: {}".format(mean_squared_error(Y_test,result_bosque)))
print("R2 del bosque de regresión: {}".format(r2_score(Y_test,result_bosque)))
print("\n\n")

# 4) Red Neuronal
parametros_red = {
    "hidden_layer_sizes":range(1,20,1),
    "alpha":np.logspace(-5,2,8)
}
red=MLPRegressor(solver='lbfgs',random_state=0)
modelo_red=GridSearchCV(red,parametros_red,cv=3)
modelo_red.fit(X_train_final,Y_train)
result_red=modelo_red.predict(X_test_final)
print("RMSE de la red neuronal: {}".format(mean_squared_error(Y_test,result_bosque)))
print("R2 del bosque de regresión: {}".format(r2_score(Y_test,result_bosque)))
print("\n\n")


"""Luego de haber corrido 4 modelos de machine learning, observamos que aquel que mejor predice la
calidad del cereal, que tiene por lejos el menor RSME y un gran R2 de 0.99 es el modelo de regresión
múltiple. Por último realizaremos un gráfico para entender visualmente el ajuste que realiza cada uno
de los modelos. """

j=1
lista=[]
for i in Y_test:
    lista.append(j)
    j=j+1
plt.scatter(lista,np.sort(Y_test))
plt.plot(lista,np.sort(result),'r-',label="Regresión Múltiple")
plt.plot(lista,np.sort(result_arbol),'b-',label="Árbol de Decisión")
plt.plot(lista,np.sort(result_bosque),'g-',label="Random Forest")
plt.plot(lista,np.sort(result_red),'y-',label="Red Neuronal")
plt.legend(loc='best')
plt.title("Comparativa de Modelos",size=20)
plt.show()
