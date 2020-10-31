
"""
En este programa se estudia la serie temporal de facturación trimestral de Amazon. Para predecir el
próximo punto de la serie se utilizan hasta 10 puntos anteriores. Se utilizan las 10 facturaciones
anteriores como variables y la facturación reciente como el resultado. Construiremos un dataset de
esa manera y luego aplicaremos diferente modelos de ML junto con el recurso de grid search para
optimizar cada modelo para encontrar la mejor regresión.
"""


#Cargamos las librerías a utilizar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ridge_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge


#Importamos el dataset a utilizar y realizamos un análisis exploratorio

amazon=pd.read_excel("data/amazon.xlsx")
print("Forma del dataset:{}".format(amazon.shape))
print("Columnas del dataset:{}".format(amazon.columns))
print("Tipos de las columnas:{}".format(amazon.dtypes))
print("Resumen de las columnas:{}".format(amazon.describe))
print("Primeros 20 valores del dataset:{}".format(amazon.head(20)))
print("Últimos 20 valores del dataset:{}".format(amazon.tail(20)))
print("\n Suma de ingresos de Amazon:{}".format(amazon['revenue'].sum()))
print("\n Suma de ganancias de Amazon:{}".format(amazon['income'].sum()))


#Realizamos algunos gráficos del dataset

plt.plot(amazon["Quarter"],amazon["revenue"],'b-')
plt.xlabel("Fechas",size=12)
plt.ylabel("Ingresos",size=12)
plt.title("Serie temporal de ingresos trimestrales",size=18)
plt.show()

plt.plot(amazon["Quarter"],amazon["income"],'r-')
plt.xlabel("Fechas",size=12)
plt.ylabel("Ganancia",size=12)
plt.title("Serie temporal de Ganancias trimestrales",size=18)
plt.show()

plt.plot(amazon["Quarter"],amazon["revenue"],'b-',label="Ingresos")
plt.plot(amazon["Quarter"],amazon["income"],'r-',label="Ganancias")
plt.legend(loc="best")
plt.xlabel("Fechas",size=12)
plt.ylabel("MM$",size=12)
plt.title("Serie temporal comparativa",size=18)
plt.show()


#Armamos nuestro nuevo dataset

vector_columna=np.asarray(amazon['revenue'])
print(vector_columna)
vector_columna=np.reshape(vector_columna,(-1,1))
filas=vector_columna.shape[0]
matriz_columna=vector_columna.copy()
print(matriz_columna)
for i in range(1,13,1):
    matriz_columna=np.concatenate((matriz_columna[1:,:],vector_columna[0:filas-i,:]),axis=1)
print(matriz_columna)
Y=matriz_columna[:,0]
X=matriz_columna[:,1:]
print("Vector Y:\n{}".format(Y))
print("Dataset X:\n{}".format(X))

"""
Ahora que ya pudimos construir nuestro nuevo dataset con las variables temporales, estamos en
condiciones de proceder a la realización de las regresiones a través de 4 modelos diferentes de
ML.
"""

# 1) Regresión Múltiple

print("\n\n\n     ****REGRESIÓN LINEAL****\n\n\n")
KF=KFold(n_splits=5, shuffle=True, random_state=123)
modelo1=LinearRegression()
parametros={}
GS1=GridSearchCV(modelo1,parametros,cv=KF)
t0=time.time()
GS1.fit(X,Y)
print("Tiempo de entrenamiento del modelo:{}".format(time.time()-t0))
print("Best score:{}".format(GS1.best_score_))
print("Error score:{}".format(GS1.error_score))
prediccion=GS1.predict(X)
plt.plot(range(Y.shape[0]),Y,'b-',label="Real")
plt.plot(range(Y.shape[0]),prediccion,'r-',label="Prediccion")
plt.legend(loc='best')
plt.title("Regresión Lineal -- Serie real vs vs Serie predicha",size=15)
plt.show()

# 2) Ridge regression

print("\n\n\n     ****RIDGE REGRESION****\n\n\n")
KF=KFold(n_splits=5, shuffle=True, random_state=123)
modelo2=Ridge()
parametros_ridge={'alpha': [.001, 0.1, 1.0, 5, 10, 50, 100, 1000]}
GS2=GridSearchCV(modelo2,parametros_ridge,cv=KF)
t1=time.time()
GS2.fit(X,Y)
print("Tiempo de entrenamiento del modelo:{}".format(time.time()-t1))
print("Best score:{}".format(GS2.best_score_))
print("Error score:{}".format(GS2.error_score))
print("Mejores parámetros:{}".format(GS2.best_params_))
prediccion2=GS2.predict(X)
plt.plot(range(Y.shape[0]),Y,'b-',label="Real")
plt.plot(range(Y.shape[0]),prediccion2,'r-',label="Prediccion")
plt.legend(loc='best')
plt.title("Ridge Regression -- Serie real vs vs Serie predicha",size=15)
plt.show()

# 3) Red neuronal

print("\n\n\n     ****RED NEURONAL****\n\n\n")
KF=KFold(n_splits=5, shuffle=True, random_state=123)
modelo3=MLPRegressor(solver='lbfgs')
parametros_red = {
    "hidden_layer_sizes":range(1,16,3),
    "alpha":np.logspace(-5,2,8)
}
GS3=GridSearchCV(modelo3,parametros_red,cv=KF)
t2=time.time()
GS3.fit(X,Y)
print("Tiempo de entrenamiento del modelo:{}".format(time.time()-t2))
print("Best score:{}".format(GS3.best_score_))
print("Error score:{}".format(GS3.error_score))
print("Mejores parámetros:{}".format(GS3.best_params_))
prediccion3=GS3.predict(X)
plt.plot(range(Y.shape[0]),Y,'b-',label="Real")
plt.plot(range(Y.shape[0]),prediccion3,'r-',label="Prediccion")
plt.legend(loc='best')
plt.title("Red neuronal -- Serie real vs vs Serie predicha",size=15)
plt.show()

# 4) ExtraTreesRegressor

print("\n\n\n     ****EXTRA TREES REGRESSOR****\n\n\n")
KF2=KFold(n_splits=4, shuffle=True, random_state=123)
modelo4=ExtraTreesRegressor(n_jobs=-1)

parametros_tree = {
    "max_depth": range(1, 7, 2),
    "min_samples_split": range(2, 14, 4),
    "min_samples_leaf":range(1,9,2),
    "n_estimators":range(2,12,2)
}
GS4=GridSearchCV(modelo4,parametros_tree,cv=KF2)
t3=time.time()
GS4.fit(X,Y)
print("Tiempo de entrenamiento del modelo:{}".format(time.time()-t3))
print("Best score:{}".format(GS4.best_score_))
print("Error score:{}".format(GS4.error_score))
print("Mejores parámetros:{}".format(GS4.best_params_))
prediccion4=GS4.predict(X)
plt.plot(range(Y.shape[0]),Y,'b-',label="Real")
plt.plot(range(Y.shape[0]),prediccion4,'r-',label="Prediccion")
plt.legend(loc='best')
plt.title("Extra Trees Regressor -- Serie real vs vs Serie predicha",size=15)
plt.show()


print("\n\n\n ****RESUMEN DE SCORES OBTENIDOS****\n\n\n")
print("Regresión lineal: {}".format(GS1.best_score_))
print("Ridge Regression: {}".format(GS2.best_score_))
print("Red Neuronal: {}".format(GS3.best_score_))
print("Extra Trees: {}".format(GS4.best_score_))


"""
Luego de generados los 4 modelos, se obseva que aquel que mejor ajusta los datos es la red
neuronal y aque que peor ajusta los datos es el algoritmo de extra trees regressor. La regresión
lineal múltiple y la ridge regression son los algoritmos de scores intermedios, con resultados muy
parecidos entre si.
"""
