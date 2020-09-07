
"""En este programa se toma una base de datos de países y se emplean 2 algoritmos que clasifiquen a
que región pertenencen dichos países de acuerdo a ciertas variables. Se comparará el accuracy de ambos modelos
y cuanto tiempo tardan en mejorar sus resultados a medida que se realiza un tuneo de sus respectivos
hiperparámetros."""


# encoding: utf-8
#Primero cargamos las librerias a utilizar

import pandas as pd
import numpy as np
import scipy
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
import sklearn
import random
import time
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


#Cargamos el dataset a utilizar y realizamos un análisis exploratorio del mismo

felicidad=pd.read_excel("https://raw.githubusercontent.com/fferrant/excel_files/master/felicidad.xls",sep=";")
print("Primeras 30 filas del dataset:")
print(felicidad.head(30))
print("Últimas 30 filas del dataset:")
print(felicidad.tail(30))
print("Variables del dataset:")
print(felicidad.columns)
print("Descripción de cada variable:")
print(felicidad.describe())
print("Descipción de alguna de las variables:")
print(felicidad["Regional indicator"].describe())
print("Promedio para alguna de las variables:")
print(felicidad["Ladder score"].mean())


#Realizamos algunos gráficos exploratorios

#Comenzamos con un scatter plot
Y=np.asarray(felicidad["Regional indicator"])
X1=np.asarray(felicidad["Healthy life expectancy"])
X2=np.asarray(felicidad["Freedom to make life choices"])
plt.plot(X2,X1,'ro')
plt.ylabel("Healthy life expectancy")
plt.xlabel("Freedom to make life choices")
plt.title("Scatter plot",size=20)
plt.show()
#Proseguimos con un boxplot
plt.figure(figsize=(20,10))
sns.boxplot(y=X1,x=Y)
plt.title("Boxplot expectativa de vida",size=25)
plt.xticks(rotation=90)
plt.show()


"""Ya concluida la etapa exploratoria del dataset, avanzamos ahora a la fase de las predicciones.
 Utilizaremos 2 algoritmos de data minning comparandolos entre sí tanto en precisión como en tiempo
 que le toma arribar a sus predicciones."""

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
X=np.asarray(felicidad.iloc[:,2:])  
print(X)
random.seed=0
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
t_arbol=0
t_bosque=0
max_arbol=0
max_bosque=0
mejor_arbol_split=0
mejor_arbol_bucket=0
mejor_arbol_depth=0
mejor_bosque_split=0
mejor_bosque_bucket=0
mejor_bosque_estimador=0
lista_arbol=[]
lista_bosque=[]
tiempo_arbol=[]
tiempo_bosque=[]

for split in range(5,15,2):
    for leaf in range(2,8,1):

        #Realizamos las iteraciones de nuestro árbol de decisión
        for depth in range (2,16,1):
            modelo1=DecisionTreeClassifier(max_depth=depth,min_samples_split=split,min_samples_leaf=leaf,random_state=0)
            pipe1=make_pipeline(StandardScaler(),modelo1)
            t0=time.time()
            pipe1.fit(X_train,Y_train)
            resultado1=pipe1.predict(X_test)
            t1=time.time()
            t_arbol=t_arbol+t1-t0
            precision1=accuracy_score(resultado1,Y_test)
            if precision1>max_arbol:
                max_arbol=precision1
                mejor_arbol_split=split
                mejor_arbol_bucket=leaf
                mejor_arbol_depth=depth
            lista_arbol.append(max_arbol)
            tiempo_arbol.append(t_arbol)

        #Realizamos las iteraciones del Random Forest
        for estimator in range (40,200,40):
            modelo2=RandomForestClassifier(n_estimators=estimator,min_samples_split=split,min_samples_leaf=leaf,n_jobs=-1,random_state=0)
            pipe2=make_pipeline(StandardScaler(),modelo2)
            t2=time.time()
            pipe2.fit(X_train,Y_train)
            resultado2=pipe2.predict(X_test)
            t3=time.time()
            t_bosque=t_bosque+t3-t2
            precision2=accuracy_score(resultado2,Y_test)
            if precision2>max_bosque:
                max_bosque=precision2
                mejor_bosque_split=split
                mejor_bosque_bucket=leaf
                mejor_bosque_estimador=estimator
            lista_bosque.append(max_bosque)
            tiempo_bosque.append(t_bosque)

        #Graficamos los resultados de ambos algoritmos
        plt.figure(figsize=(30,10))
        plt.plot(tiempo_arbol,lista_arbol,'b-',label="Decission Tree")
        plt.plot(tiempo_bosque,lista_bosque,'r-',label="Random Forest")
        plt.title("Precisión del modelo en función del tiempo")
        plt.xlabel("Tiempo")
        plt.ylabel("Accuracy")
        plt.legend(loc='best')
        plt.savefig("grafico.png")


#Vamos a imprimir ahora los hiperparámetros que mejor predicen los datos para uno y otro modelo

# 1A) Para el árbol de decisión
print("Mejores hiperparámetros del árbol de decisión:\n")
print("Mejor max_depth: {}".format(mejor_arbol_depth))
print("Mejor min_samples_split: {}".format(mejor_arbol_split))
print("Mejor min_samples_leaf: {}".format(mejor_arbol_bucket))

# 2A) Para el Random Forest
print("Mejores hiperparámetros del Random Forest:\n")
print("Mejor n_estimators: {}".format(mejor_bosque_estimador))
print("Mejor min_samples_split: {}".format(mejor_bosque_split))
print("Mejor min_samples_leaf: {}".format(mejor_bosque_bucket))


"""Finalmente con los hiperparámetros que mejor ajustan para cada uno de los algoritmos, obtendremos
todas las métricas posibles: no sólo el accuracy, sino además la matriz de confusión y el área debajo
de la curva"""


# 1B) Para el árbol de decisión
modelo_arbol=DecisionTreeClassifier(max_depth=mejor_arbol_depth,min_samples_split=mejor_arbol_split,min_samples_leaf=mejor_arbol_bucket,random_state=0)
pipe3=make_pipeline(StandardScaler(),modelo_arbol)
pipe3.fit(X_train,Y_train)
resultado_arbol=pipe3.predict(X_test)
print("\n Métricas del mejor árbol de clasificación:\n")
print("Accuracy: {}".format(accuracy_score(resultado_arbol,Y_test)))
print("Matriz de confusión del modelo:\n {}".format(confusion_matrix(Y_test,resultado_arbol)))
print("Score de la curva ROC: {}\n".format(roc_auc_score(Y_test,pipe3.predict_proba(X_test),multi_class='ovo')))

# 2B) Para el Random Forest
modelo_bosque=RandomForestClassifier(n_estimators=mejor_bosque_estimador,min_samples_split=mejor_bosque_split,min_samples_leaf=mejor_bosque_bucket,random_state=0)
pipe4=make_pipeline(StandardScaler(),modelo_bosque)
pipe4.fit(X_train,Y_train)
resultado_bosque=pipe4.predict(X_test)
print("\n Métricas del mejor Random Forest:\n")
print("Accuracy: {}".format(accuracy_score(resultado_bosque,Y_test)))
print("Matriz de confusión del modelo:\n {}".format(confusion_matrix(Y_test,resultado_bosque)))
print("Score de la curva ROC: {}\n".format(roc_auc_score(Y_test,pipe4.predict_proba(X_test),multi_class='ovo')))

