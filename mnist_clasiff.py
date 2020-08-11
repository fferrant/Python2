

"""
En este programa utilizaremos diferentes algoritmos de clasificación para predecir a que dígito pertenecen cada
uno de los vectores imágenes del dataset MNIST. Los algoritmos que usaremos para predecir son: árboles de
decisión, random forest, SVM, naive bayes por último una red neuronal con una capa oculta. Una vez generados los
 modelos, procederemos a obtener métricas de bondad de cada uno de ellos: score, matriz de confusión y área debajo
de la curva.
"""


#Importamos las librerías a utilizar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import random
import time
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_confusion_matrix
from sklearn import svm
from sklearn.naive_bayes import GaussianNB


#Carga y análisis exploratorio del dataset a utilizar
X,y= load_digits(return_X_y=True)
print("\n Dimensiones del dataset:\n")
print(X.shape)
print("\n Dimensiones del vector solución:\n")
print(y.shape)
print("\n Cantidad de imágenes de cada dígito:\n")
print(np.unique(y,return_counts=True)) #Se ve que las clases están muy balanceadas (entre 174 y 183 ncounts)


# Generamos ahora un split de datos entre train y test
random.seed=0
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


#Ahora que tenemos nuestros datasets de train y test, comencemos con nuestros algoritmos de clasificación

# 1) Árbol de decisión
params={"max_depth":range(1,9,2),
        "min_samples_split":range(6,30,6),
        "min_samples_leaf":range(3,15,3)}
random.seed=0
arbol=DecisionTreeClassifier(random_state=0)
t0=time.time()
modelo=GridSearchCV(arbol,params,cv=6).fit(X_train,y_train) #Hasta acá es el armado del modelo
print("\n Tiempo tomado para generar el modelo: {}\n".format(time.time()-t0))
resultado=modelo.predict(X_test) #Realizamos predicciones a partir del modelo
print("Primeros resultados obtenidos:\n")
print(resultado[0:20])
print("\n\n")
tabla=pd.DataFrame(data=modelo.cv_results_) #Aca comenzamos con la evaluación del modelo
print(tabla)
print("Mejores parámetros:{}\n".format(modelo.best_params_)) 
print("Mejor modelo: {}\n".format(modelo.best_estimator_))
print("Score del mejor modelo: {}\n".format(modelo.best_score_))
print("Precisión del modelo:{}\n".format(accuracy_score(y_test,resultado)))
print("Matriz de confusión del modelo:\n {}\n".format(confusion_matrix(y_test,resultado)))
print("Score de la curva ROC: {}\n".format(roc_auc_score(y_test,modelo.predict_proba(X_test),multi_class='ovo')))


# 2) Random Forest
params2={"max_depth":range(1,7,2),
        "min_samples_split":range(6,18,6),
        "min_samples_leaf":range(3,12,3),
        "n_estimators":range(10,30,10)}
random.seed=0
bosque=RandomForestClassifier(random_state=0)
t1=time.time()
modelo2=GridSearchCV(bosque,params2,cv=4).fit(X_train,y_train) #Hasta acá es el armado del modelo
print("\n Tiempo tomado para generar el modelo: {}\n".format(time.time()-t1))
resultado2=modelo2.predict(X_test) #Realizamos predicciones a partir del modelo
print("Primeros resultados obtenidos:\n")
print(resultado2[0:20])
print("\n\n")
tabla2=pd.DataFrame(data=modelo2.cv_results_) #Aca comenzamos con la evaluación del modelo
print(tabla2)
print("Mejores parámetros:{}\n".format(modelo2.best_params_)) 
print("Mejor modelo: {}\n".format(modelo2.best_estimator_))
print("Score del mejor modelo: {}\n".format(modelo2.best_score_))
print("Precisión del modelo:{}\n".format(accuracy_score(y_test,resultado2)))
print("Matriz de confusión del modelo:\n {}\n".format(confusion_matrix(y_test,resultado2)))
print("Score de la curva ROC, método OvO: {}\n".format(roc_auc_score(y_test,modelo2.predict_proba(X_test),multi_class='ovo')))
print("Score de la curva ROC, método OvR: {}\n".format(roc_auc_score(y_test,modelo2.predict_proba(X_test),multi_class='ovr')))
disp = plot_confusion_matrix(modelo2, X_test, y_test,cmap=plt.cm.Blues,display_labels=np.unique(y))
disp.ax_.set_title("Confusion Matrix Plot RF")
plt.gca().invert_yaxis()
plt.show()


# 3) Support Vector Machine
params3={"C":range(1,5,1),
         "gamma":range(1,5,1),
         "kernel":['rbf','linear']}
random.seed=0
soporte=svm.SVC(random_state=0)
t2=time.time()
modelo3=GridSearchCV(soporte,params3,cv=5).fit(X_train,y_train) #Hasta acá es el armado del modelo
print("\n Tiempo tomado para generar el modelo: {}\n".format(time.time()-t2))
resultado3=modelo3.predict(X_test) #Realizamos predicciones a partir del modelo
print("Primeros resultados obtenidos:\n")
print(resultado3[0:20])
print("\n\n")
tabla3=pd.DataFrame(data=modelo3.cv_results_) #Aca comenzamos con la evaluación del modelo
print(tabla3)
print("Mejores parámetros:{}\n".format(modelo3.best_params_)) 
print("Mejor modelo: {}\n".format(modelo3.best_estimator_))
print("Score del mejor modelo: {}\n".format(modelo3.best_score_))
print("Precisión del modelo:{}\n".format(accuracy_score(y_test,resultado3)))
print("Matriz de confusión del modelo:\n {}\n".format(confusion_matrix(y_test,resultado3)))
disp = plot_confusion_matrix(modelo3, X_test, y_test,cmap=plt.cm.Blues,display_labels=np.unique(y))
disp.ax_.set_title("Confusion Matrix Plot SVM")
plt.gca().invert_yaxis()
plt.show()


# 4) Naive Bayes
random.seed=0
bayesian = GaussianNB()
t4=time.time()
modelo5=bayesian.fit(X_train,y_train)
print("\n Tiempo tomado para generar el modelo: {}\n".format(time.time()-t4))
resultado5=modelo5.predict(X_test) #Realizamos predicciones a partir del modelo
print("Primeros resultados obtenidos:\n")
print(resultado5[0:20])
print("\n\n")
print("Precisión del modelo:{}\n".format(accuracy_score(y_test,resultado5)))
print("Matriz de confusión del modelo:\n {}\n".format(confusion_matrix(y_test,resultado5)))
print("Score de la curva ROC, método OvO: {}\n".format(roc_auc_score(y_test,modelo5.predict_proba(X_test),multi_class='ovo')))
print("Score de la curva ROC, método OvR: {}\n".format(roc_auc_score(y_test,modelo5.predict_proba(X_test),multi_class='ovr')))
disp = plot_confusion_matrix(modelo5, X_test, y_test,cmap=plt.cm.Blues,display_labels=np.unique(y))
disp.ax_.set_title("Confusion Matrix Plot NB")
plt.gca().invert_yaxis()
plt.show()


# 5) Red Neuronal
params4={"hidden_layer_sizes":range(1,30,1)}
random.seed=0
red=MLPClassifier(random_state=0)
t3=time.time()
modelo4=GridSearchCV(red,params4,cv=8).fit(X_train,y_train) #Hasta acá es el armado del modelo
print("\n Tiempo tomado para generar el modelo: {}\n".format(time.time()-t3))
resultado4=modelo4.predict(X_test) #Realizamos predicciones a partir del modelo
print("Primeros resultados obtenidos:\n")
print(resultado4[0:20])
print("\n\n")
tabla4=pd.DataFrame(data=modelo4.cv_results_) #Aca comenzamos con la evaluación del modelo
print(tabla4)
print("Mejores parámetros:{}\n".format(modelo4.best_params_)) 
print("Mejor modelo: {}\n".format(modelo4.best_estimator_))
print("Score del mejor modelo: {}\n".format(modelo4.best_score_))
print("Precisión del modelo:{}\n".format(accuracy_score(y_test,resultado2)))
print("Matriz de confusión del modelo:\n {}\n".format(confusion_matrix(y_test,resultado4)))
print("Score de la curva ROC, método OvO: {}\n".format(roc_auc_score(y_test,modelo4.predict_proba(X_test),multi_class='ovo')))
print("Score de la curva ROC, método OvR: {}\n".format(roc_auc_score(y_test,modelo4.predict_proba(X_test),multi_class='ovr')))
disp = plot_confusion_matrix(modelo4, X_test, y_test,cmap=plt.cm.Blues,display_labels=np.unique(y))
disp.ax_.set_title("Confusion Matrix Plot NN")
plt.gca().invert_yaxis()
plt.show()
