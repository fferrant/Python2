


#Importamos las librerías a utilizar
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from scipy import stats
import seaborn as sns
import pandas as pd
import missingno as msno
import sklearn
import time


#Cargamos el dataset y obtenemos algunos valores del mismo
bomba=pd.read_csv("data/sensor.csv",sep=",")
print("\n Primeros 5 valores del dataset:\n")
print(bomba.head(5))
print("\n Últimos 5 valores del dataset:\n")
print(bomba.tail(5))
print("\n Variables del dataset:\n")
print(bomba.columns)
print("\n Resumen de cada variable\n")
print(bomba.describe())
bomba2=bomba.groupby("machine_status")
print("\n Resumen de cada variable abierta por máquina:\n")
print(bomba2.describe())
print("\n Forma del dataset:\n")
print(bomba.shape)
print("\n Promedio de cada feature por status de máquina:\n")
print("\n Para broken:\n")
print(bomba[bomba["machine_status"]=="BROKEN"].mean())
print("\n Resumen detallado de cada variable:\n")
print(bomba.iloc[:,0:10].describe())
print(bomba.iloc[:,11:20].describe())
print(bomba.iloc[:,21:30].describe())
print(bomba.iloc[:,31:40].describe())
print(bomba.iloc[:,41:50].describe())

#Análisis y tratamiento de valores nulos
msno.bar(bomba)
plt.show()
msno.heatmap(bomba.iloc[:,0:10])
plt.show()
msno.heatmap(bomba.iloc[:,11:20])
plt.show()
msno.heatmap(bomba.iloc[:,21:30])
plt.show()
msno.heatmap(bomba.iloc[:,31:40])
plt.show()
msno.heatmap(bomba.iloc[:,41:50])
plt.show() #Los sensores 1,6,7,14,15,16,17,22,25,26,29,30 y 32 poseen gran cantidad de nulos
de_nulos=bomba[pd.isnull(bomba).any(axis=1)]
print("\n Cantidad de filas con algún valor nulo:\n")
print(de_nulos["sensor_00"].count()) 


#Eliminación de columnas con gran cantidad de valores nulos
bomba=bomba.drop(["sensor_00","sensor_01","sensor_06","sensor_07","sensor_14","sensor_15","sensor_16"],axis=1)
bomba=bomba.drop(["sensor_17","sensor_22","sensor_25","sensor_26","sensor_29","sensor_30","sensor_32"],axis=1)
print("\nColumnas del nuevo dataset:\n")
print(bomba.columns)
de_nulos2=bomba[pd.isnull(bomba).any(axis=1)]
print("\n Cantidad de filas con algún valor nulo:\n")
print(de_nulos2["sensor_05"].count())
print("\n Cantidad de nulos por columna:\n")
print(bomba[pd.isnull(bomba).any(axis=1)].count())
bomba=bomba.dropna()
print("Forma final de nuestro dataset:\n")
print(bomba.shape)


#Generación de gráficos que representen la serie temporal
bomba["binario"]=0
bomba.loc[bomba["machine_status"]=="NORMAL",["binario"]]=1
print(bomba["binario"].describe)
print(bomba["binario"].sum())
plt.plot(bomba["binario"])
plt.xlabel("Fecha y hora",size=15)
plt.ylabel("Estado",size=15)
plt.title("Serie temporal del estado de una bomba",size=25)
plt.show() # Graficaremos a continuación las mediciones de algunos sensores:
sns.boxplot(y=bomba["sensor_02"],x=np.full(len(bomba["sensor_02"]),"sensor_02")) #Gráficos de cajas
plt.title("Boxplot sensor 2")
plt.show()
sns.boxplot(y=bomba["sensor_20"],x=np.full(len(bomba["sensor_20"]),"sensor_20"))
plt.title("Boxplot sensor 20")
plt.show()
sns.boxplot(y=bomba["sensor_35"],x=np.full(len(bomba["sensor_35"]),"sensor_35"))
plt.title("Boxplot sensor 35")
plt.show()
sns.boxplot(y=bomba["sensor_44"],x=np.full(len(bomba["sensor_44"]),"sensor_44"))
plt.title("Boxplot sensor 44")
plt.show()
plt.plot(bomba["sensor_02"],label="Sensor 2")
plt.plot(bomba["sensor_20"],label="Sensor 20")
plt.plot(bomba["sensor_35"],label="Sensor 35")
plt.plot(bomba["sensor_44"],label="Sensor 44")
plt.title("Serie temporal de varios sensores",size=25)
plt.xlabel("Tiempo",size=15)
plt.ylabel("Valor",size=15)
plt.legend(loc="best")
plt.show()


"""
A continuación graficaremos el valor arrojado por cada uno de los sensores vs el status de la bomba representado
de forma binaria, en la cual 1 representa la máquina en fucionamiento y 0 representa la máquina inactiva. Para
que las mediciones de los sensores sean comparables contra el status de la bomba, primero debemos normalizarlas.
Con el objetivo de poder testear varios sensores vs el status de máquina de forma rápida, generamos una fución
cuyo argumento sea la variable a comparar vs el statu de máquina.
"""


def comparar(variable):
    variable=str(variable)
    normal=np.asarray(bomba[variable])
    normal=(normal-bomba[variable].mean())/bomba[variable].std()
    plt.plot(bomba["binario"],label="Estado de la bomba")
    plt.plot(normal,label="Variable {}".format(variable))
    plt.legend(loc="best")
    plt.xlabel("Tiempo",size=15)
    plt.ylabel("Valores normalizados",size=15)
    plt.title("Status de la bomba en función de la variable {}".format(variable),size=20)
    plt.annotate("Para el estado de la bomba 1 representa un estado activo y 0 un estado inactivo",xy=(10,-2),size=10)
    plt.show()
comparar("sensor_02") #Parece correlacionar muy bien con las fallas de la bomba
comparar("sensor_20") #Parece NO correlacionar con las fallas de la bomba
comparar("sensor_35") #Parece NO correlacionar con las fallas de la bomba
comparar("sensor_44") #Parece NO correlacionar con las fallas de la bomba


"""
Luego de la etapa exploratoria de nuestro dataset, de haber limpiado los datos y de haber realizado algunas
visualizaciones de las series temporales tanto del estado de la bomba com así también de features que podrían 
o no explicar el fenómeno, procederemos a realizar algunos modelos de regresión y machine learning que nos permitan 
predecir una falla.
"""
print(bomba.columns)

#Importemos las herramientas de ML con las que vamos a trabajar
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_roc_curve

#Vamos ahora a realizar y evaluar distintos modelos para predecir y clasificar correctamente el estado de la bomba
#Empecemos con algunas jugadas comunes a todos los modelos
X=bomba.iloc[:,2:39].to_numpy()
y=bomba.iloc[:,40].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(y_test[0:75])
print(y_train[0:75])
print(X_test[0:75])
print(X_train[0:75])


#Ahora si empezamos con los modelos de clasificación. 1) Regresión logística
pipe=make_pipeline(StandardScaler(),LogisticRegression(random_state=0))
pipe.fit(X_train,y_train)
resultado=pipe.predict(X_test)
print("\n Regresión Logística \n")
print("\n Predicción de los primeros 20 valores:\n")
print(resultado[0:20])
print("\Precisión del modelo obtenido:\n")
print(accuracy_score(pipe.predict(X_test),y_test))
print("\n Matriz de confusión del modelo:\n")
print(confusion_matrix(y_test,pipe.predict(X_test),labels=['RECOVERING','NORMAL','BROKEN']))


#2) Linear discriminant analysis
pipe2=make_pipeline(StandardScaler(),LinearDiscriminantAnalysis())
pipe2.fit(X_train,y_train)
resultado2=pipe2.predict(X_test)
print("\n Linear discriminant analysis \n")
print("\n Predicción de los primeros 20 valores:\n")
print(resultado2[0:20])
print("\Precisión del modelo obtenido:\n")
print(accuracy_score(pipe2.predict(X_test),y_test))
print("\n Matriz de confusión del modelo:\n")
print(confusion_matrix(y_test,pipe2.predict(X_test),labels=['RECOVERING','NORMAL','BROKEN']))


# 3) Decision Tree
arbol=DecisionTreeClassifier(random_state=0)
parameters2 = {
    "max_depth": range(1, 7, 2),
    "min_samples_split": range(2, 19, 2)
}
pipe3=make_pipeline(StandardScaler(),GridSearchCV(arbol,parameters2,cv=3))
t0=time.time()
pipe3.fit(X_train,y_train)
print("\n Tiempo demandado en generar el modelo:{} secs\n".format(time.time()-t0))
resultado3=pipe3.predict(X_test)
print("\n Árbol de decisión \n")
print("\n Predicción de los primeros 20 valores:\n")
print(resultado3[0:20])
print("\Precisión del modelo obtenido:\n")
print(accuracy_score(pipe3.predict(X_test),y_test))
print("\n Matriz de confusión del modelo:\n")
print(confusion_matrix(y_test,pipe3.predict(X_test),labels=['RECOVERING','NORMAL','BROKEN']))


# 4) Random Forest
forest=RandomForestClassifier(n_estimators=100,random_state=0)
parameters = {
    "max_depth": range(1, 7, 2),
    "min_samples_split": range(2, 19, 2)
}
pipe4=make_pipeline(StandardScaler(),GridSearchCV(forest,parameters,cv=3))
t0=time.time()
pipe4.fit(X_train,y_train)
print("\n Tiempo demandado en generar el modelo:{} secs\n".format(time.time()-t0))
resultado4=pipe4.predict(X_test)
print("\n Random Forest \n")
print("\n Predicción de los primeros 20 valores:\n")
print(resultado4[0:20])
print("\Precisión del modelo obtenido:\n")
print(accuracy_score(pipe4.predict(X_test),y_test))
print("\n Matriz de confusión del modelo:\n")
print(confusion_matrix(y_test,pipe4.predict(X_test),labels=['RECOVERING','NORMAL','BROKEN']))
