
""" En este programa se realizará una reducción de dimensiones de un dataset de países y sus parámetros
de felicidad a tráves de PCA. A partir de los 2 componentes principales hallados se tratará "adivinar"
los diferentes custers que forman el dataset. Luego se compararán esos cluster "adivinados" con un 
análisis K-means con igual cantidad de clusters propuestos. También se compará este análisis con la
región a la que pertenece cada país estudiado."""


# encoding: utf-8
#Primero cargamos las librerias a utilizar

import pandas as pd
import numpy as np
import scipy
import matplotlib
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
from sklearn import decomposition
from sklearn.cluster import KMeans


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


#A partir del dataset cargado, obtenemos los 2 componentnes principales y los gráficamos
X=felicidad.iloc[:,2:]
Y=felicidad.iloc[:,:2]
Region=np.asarray(felicidad["Regional indicator"])
Region=Region.reshape(-1,1)
pca=decomposition.PCA(n_components=2)
X_pca=pca.fit_transform(X)
X_pca=np.hstack((X_pca,Region))
plt.scatter(X_pca[:,0],X_pca[:,1])
plt.xlabel("Primer componente principal",size=12)
plt.ylabel("Segundo componente principal",size=12)
plt.title("Reducción de dimensiones con PCA",size=20)
plt.show()

#Volvamos a imprimir el scatter plot, pero identificando a cada región por separado
for i in felicidad["Regional indicator"].unique():
    plt.scatter(X_pca[:,0][X_pca[:,2]==i],X_pca[:,1][X_pca[:,2]==i],label=i)
plt.legend(loc="best")
plt.xlabel("Primer componente principal",size=12)
plt.ylabel("Segundo componente principal",size=12)
plt.title("Reducción de dimensiones con PCA",size=20)
plt.show()

#Repetimos el gráfico sin leyenda para poder observar toda la grilla:
for i in felicidad["Regional indicator"].unique():
    plt.scatter(X_pca[:,0][X_pca[:,2]==i],X_pca[:,1][X_pca[:,2]==i],label=i)
plt.xlabel("Primer componente principal",size=12)
plt.ylabel("Segundo componente principal",size=12)
plt.title("Reducción de dimensiones con PCA",size=20)
plt.show()


""" A partir de los últimos 2 gráficos trazados, se observa que la primer componente principal es
una mucho mejor variable explicadora de las diferentes regiones que la segunda componente principal.
A grandes rasgos podríamos decir que los países de las regiones 'Western Europe' y 'North America'
tienen valores de primer componente principal de entre -infinito y -7; la región 'Sub-Saharan Africa
tiene valores de primer componente principal entre 5 y +infinito; mientras que las 7 regiones restantes
tienen valores de primer componente principal entre -7 y 5. A continuación vamos a realizar 3 gráficos
más: haremos 2 boxplots por región una a lo largo de la primer componente principal y otra a lo largo
de la segunda componente principal para validar nuestra hipótesis de que los datos están mucho más
concentrados y con menos varianza a lo largo de la primer componente que respecto a la segunda. El
otro gráfico que haremos será un scatter plot como los vistos más arriba, pero que separe los datos
en 3 clusters según los 3 grandes intervalos de 1er componente principal hallados. Más tarde obtendremos
3 clusters pero en este caso a partir de un algoritmo de K-Means y compararemos esos nuevos clusters con
esta primer propuesta que realizamos a partir de la mera observación."""


plt.subplot(211)
sns.boxplot(x=X_pca[:,2],y=X_pca[:,0])
plt.tick_params(axis='x',which='both',bottom=False, top=False,labelbottom=False)
plt.ylabel("Primer componente")
plt.subplot(212)
sns.boxplot(x=X_pca[:,2],y=X_pca[:,1])
plt.suptitle("Boxplot componentes principales",size=18)
plt.tick_params(axis='x',which='both',bottom=False, top=False,labelbottom=False)
plt.ylabel("Segunda componente")
plt.show() #El gráfico con los 2 boxplots uno debajo del otro comprueba nuestra hipótesis

#Tracemos ahora el scatter plot con los 3 cluster propuestos

plt.scatter(X_pca[:,0][X_pca[:,0]<-7],X_pca[:,1][X_pca[:,0]<-7],color="blue",label="Cluster 1")
plt.scatter(X_pca[:,0][X_pca[:,0]>5],X_pca[:,1][X_pca[:,0]>5],color="red",label="Cluster 3")
plt.scatter(X_pca[:,0][(X_pca[:,0]>-7) & (X_pca[:,0]<5)],X_pca[:,1][(X_pca[:,0]>-7) & (X_pca[:,0]<5)],color="green",label="Cluster 2")
plt.legend(loc="best")
plt.title("Clusters propuestos a partir de PCA",size=15)
plt.xlabel("Primer componente principal",size=10)
plt.ylabel("Segundo componente principal",size=10)
plt.axvline(x=-7)
plt.axvline(x=5)
plt.show()


"""Ahora que ya tenemos una propuesta de 3 cluster fruto de la observación de los componentes principales,
vamos a continuación a encontrar 3 cluster de una forma analítica esta vez a partir del algoritmo de K-means
y del dataset original."""


X_matriz=np.asarray(X)
np.random.seed(0) #Colocamos una semilla para que el experimento sea replicable
modelo=KMeans(n_clusters=3,init='random').fit(X_matriz)
resultado=modelo.predict(X_matriz)
print("\n\n\n\n Imprimimos algunos parámetros del modelo:")
print("\n\n Cantidad de clusters: {}".format(modelo.n_clusters))
print("Coeficioente de Silhouette: %0.3f" % metrics.silhouette_score(X_matriz,resultado))
print(" Centros de cada uno de los clusters: {}".format(modelo.cluster_centers_))
print("Clusters obtenidos:{}".format(resultado))


""" Ya con la clusterización por K-means realizada, avanzamos hacia el último paso: comparamos los
agrupamientos realizados por PCA + observación con los agrupamientos realizados por K-Means."""


fila_cluster=resultado.reshape(-1,1)
X_pca=np.hstack((X_pca,fila_cluster))
#Primer subgráfico con PCA
plt.subplot(211)
plt.scatter(X_pca[:,0][X_pca[:,0]<-7],X_pca[:,1][X_pca[:,0]<-7],color="blue",label="Cluster 1")
plt.scatter(X_pca[:,0][X_pca[:,0]>5],X_pca[:,1][X_pca[:,0]>5],color="red",label="Cluster 3")
plt.scatter(X_pca[:,0][(X_pca[:,0]>-7) & (X_pca[:,0]<5)],X_pca[:,1][(X_pca[:,0]>-7) & (X_pca[:,0]<5)],color="green",label="Cluster 2")
plt.ylabel("Clustering por PCA",size=10)
plt.legend(loc="best")
#Segundo subgráfico
plt.subplot(212)
plt.scatter(X_pca[:,0][X_pca[:,3]==0],X_pca[:,1][X_pca[:,3]==0],color="orange",label="Cluster 4")
plt.scatter(X_pca[:,0][X_pca[:,3]==1],X_pca[:,1][X_pca[:,3]==1],color="violet",label="Cluster 5")
plt.scatter(X_pca[:,0][X_pca[:,3]==2],X_pca[:,1][X_pca[:,3]==2],color="lightblue",label="Cluster 6")
plt.ylabel("Clustering por K-Means",size=10)
plt.suptitle("Comparación de clustering por 2 métodos",size=15)
plt.legend(loc="best")
plt.show()

#Hallamos que los agrupamientos formados por 2 diferentes métodos son muy parecidos!!!
