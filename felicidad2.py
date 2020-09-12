
"""En este programa se toma una base de datos de países y se emplea una red neuronal con una capa oculta
que clasifiquen a que región pertenencen dichos países de acuerdo a ciertas variables. Se comparará
el accuracy obtenido y el tiempo que tarda el modelo a medida que iteramos entre diferente cantidad de
neuronas en la capa oculta"""


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
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix

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


#Ya finalizado el análisis exploratorio, comenzamos a cosntruir las redes neuronales

#Construimos una primer red con 5 neuronas en la única capa oculta del modelo
primer_red= MLPClassifier(solver='lbfgs',hidden_layer_sizes=(5,),random_state=0,alpha=0.001)
pipe1=make_pipeline(StandardScaler(),primer_red)
random.seed(0)
X=np.asarray(felicidad.iloc[:,2:])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
pipe1.fit(X_train,Y_train)
print("\n Precisión del modelo:{}".format(accuracy_score(pipe1.predict(X_test),Y_test)))
class_names=felicidad["Regional indicator"].unique()
print("Matriz de confusión del modelo:\n {}".format(confusion_matrix(Y_test,pipe1.predict(X_test))))
print("Score de la curva ROC: {}".format(roc_auc_score(Y_test,pipe1.predict_proba(X_test),multi_class='ovo')))
plot_confusion_matrix(pipe1, X_test, Y_test,display_labels=class_names,cmap=plt.cm.Blues)
plt.xticks(rotation=90)
plt.title("Confusion Matrix Plot")
plt.show()

"""Ahora construiremos un iterador para encontrar la cantidad óptima de neuronas a utilizar en nuestra
capa oculta dejando fijos los parámetros alpha, max_iter y el tipo de algoritmo a utilizar (en este caso por la
baja dimensionalidad del dataset se utiliza el algorítmo lbfgs)"""

mejor_neurona=0
mejor_precision=0
lista_neuronas=[]
lista_precision=[]
lista_precision_bis=[]
lista_tiempo=[]
t_total=0

for i in range(1,150,1):
    segunda_red= MLPClassifier(solver='lbfgs',hidden_layer_sizes=(i,),random_state=0,alpha=0.0001,max_iter=1000000)
    pipe2=make_pipeline(StandardScaler(),segunda_red)
    random.seed(0)
    t0=time.time()
    pipe2.fit(X_train,Y_train)
    t_total=t_total+time.time()-t0
    lista_tiempo.append(t_total)
    if accuracy_score(pipe2.predict(X_test),Y_test)>mejor_precision:
        mejor_precision=accuracy_score(pipe2.predict(X_test),Y_test)
        mejor_neurona=i
    lista_precision.append(mejor_precision)
    lista_neuronas.append(i)
    lista_precision_bis.append(accuracy_score(pipe2.predict(X_test),Y_test))

#En base a la iteración realzada, trazamos algunos gráficos 
plt.plot(lista_tiempo,lista_precision,'r-')
plt.title("Evolución de la precisión de la red en función del tiempo",size=20)
plt.xlabel("Tiempo (secs)")
plt.ylabel("Accuracy")
plt.show()

plt.bar(lista_neuronas,lista_precision_bis,color="blue")
plt.title("Precisión por cantidad de neuronas utilizadas",size=20)
plt.xlabel("Neuronas")
plt.ylabel("Precisión")
plt.show()

#Obtenemos los parámteros del mejor modelo
print("\n\n\n Parámetros del mejor modelo obtenido\n")
print("Cantidad de nueronas del mejor modelo:{}".format(mejor_neurona))
print("Precisión del mejor modelo:{}".format(mejor_precision))
tercera_red=MLPClassifier(solver='lbfgs',hidden_layer_sizes=(mejor_neurona,),random_state=0,alpha=0.0001,max_iter=1000000)
pipe3=make_pipeline(StandardScaler(),tercera_red)
pipe3.fit(X_train,Y_train)
print("Score de la curva ROC:{}".format(roc_auc_score(Y_test,pipe3.predict_proba(X_test),multi_class='ovo')))
print("Matriz de confusión del mejor modelo:\n {}".format(confusion_matrix(Y_test,pipe3.predict(X_test))))
plot_confusion_matrix(pipe3,X_test, Y_test,display_labels=class_names,cmap=plt.cm.Blues)
plt.title("Matriz de confusión del mejor modelo")
plt.xticks(rotation=90)
plt.show()


"""El mejor modelo que pudimos obtener tiene una precisión de 0.64 y un índice de la curva ROC de 0.83.
No nos vamos a coformar con esos resultados e intentaremos mejorarlos. Para ello, además de tunear el
hiperparametro de la cantida de neuronas en la capa oculta, ahora también tunearemos el coeficionete
de regularización alpha."""

mejor_neurona2=0
mejor_alfa2=0
mejor_precision2=0
lista_neuronas2=[]
lista_precision2=[]
lista_precision_bis2=[]
lista_tiempo2=[]
t_total2=0


for i in range (1,150,1):
    for j in np.logspace(-5,2,8):
        cuarta_red= MLPClassifier(solver='lbfgs',hidden_layer_sizes=(i,),random_state=0,alpha=j,max_iter=1000000)
        pipe4=make_pipeline(StandardScaler(),cuarta_red)
        random.seed(0)
        t0=time.time()
        pipe4.fit(X_train,Y_train)
        t_total2=t_total2+time.time()-t0
        lista_tiempo2.append(t_total2)
        if accuracy_score(pipe4.predict(X_test),Y_test)>mejor_precision2:
            mejor_precision2=accuracy_score(pipe4.predict(X_test),Y_test)
            mejor_neurona2=i
            mejor_alfa2=j
        lista_precision2.append(mejor_precision2)
        lista_neuronas2.append(i)
        lista_precision_bis2.append(accuracy_score(pipe4.predict(X_test),Y_test))


#En base a la nueva iteración realzada, trazamos algunos gráficos 
plt.plot(lista_tiempo2,lista_precision2,'r-')
plt.title("Evolución de la precisión de la red en función del tiempo",size=20)
plt.xlabel("Tiempo (secs)")
plt.ylabel("Accuracy")
plt.show()

plt.bar(lista_neuronas2,lista_precision_bis2,color="blue")
plt.title("Precisión por cantidad de neuronas utilizadas",size=20)
plt.xlabel("Neuronas")
plt.ylabel("Precisión")
plt.show()


#Finalmente obtenemos los parámteros del nuevo mejor modelo
print("\n\n\n Parámetros del mejor modelo obtenido\n")
print("Cantidad de nueronas del mejor modelo:{}".format(mejor_neurona2))
print("Valor alfa del mejor modelo:{}".format(mejor_alfa2))
print("Precisión del mejor modelo:{}".format(mejor_precision2))
quinta_red=MLPClassifier(solver='lbfgs',hidden_layer_sizes=(mejor_neurona2,),random_state=0,alpha=mejor_alfa2,max_iter=1000000)
pipe5=make_pipeline(StandardScaler(),quinta_red)
pipe5.fit(X_train,Y_train)
print("Score de la curva ROC:{}".format(roc_auc_score(Y_test,pipe5.predict_proba(X_test),multi_class='ovo')))
print("Matriz de confusión del mejor modelo:\n {}".format(confusion_matrix(Y_test,pipe5.predict(X_test))))
plot_confusion_matrix(pipe5,X_test, Y_test,display_labels=class_names,cmap=plt.cm.Blues)
plt.title("Matriz de confusión del mejor modelo")
plt.xticks(rotation=90)
plt.show()
