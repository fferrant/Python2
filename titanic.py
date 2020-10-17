""" En este programa utilizaremos diferentes modelos de ML para predecir que pasjeros de Titanic 
sobrevivieron y quienes no. La estrtegia con el dataset que tenemos será no solamente la de probar
diferentes algoritmos tuneando hiperparametros, sino además hacer algo de feature Engineeering al
crear nuevas variables"""


# En primer lugar cargamos las librerías que vamos a necesitar

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
import random
import time
import math

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


#Cargamos nuestro dataset y realizamos un análisis exploratorio del mismo

titanic=pd.read_csv("data/train.csv",sep=',')
print(titanic.head(30))
print("Variables del dataset:{}".format(titanic.columns))
print("Forma del dataset: {}".format(titanic.shape))
print("Descripción de las variables: {}".format(titanic.describe))
#Separamos las variables de los resultados
X=titanic.iloc[:,2:]
Y=titanic.iloc[:,1]
print(X.head(30))
print(Y.head(30))
X.pop("Name")
X.pop("Ticket")
print(X.head(30))
semillas=[1,43,37,33,919,1040,36,2134,1000,1441,7,22,39,489,27,71,563,9999,5,7,98,12,1009,908,56,43,301,2,88,8]

#Un poco de fetaure engeneering jugando con la demografía

X["demograf"]="NA"
X.loc[(X["Age"]<=18) & (X["Sex"]=="female"),"demograf"]="Niña"
X.loc[(X["Age"]<=18) & (X["Sex"]=="male"),"demograf"]="Niño"
X.loc[(X["Age"]>=65) & (X["Sex"]=="female"),"demograf"]="Anciana"
X.loc[(X["Age"]>=65) & (X["Sex"]=="male"),"demograf"]="Anciano"
X.loc[(X["Age"]<65) & (X["Age"]>18) & (X["Sex"]=="female"),"demograf"]="Adulta"
X.loc[(X["Age"]<65) & (X["Age"]>18) & (X["Sex"]=="male"),"demograf"]="Adulto"
print(X["demograf"].describe())


#Procedemos a preprocesar el dataset

num_cols=["Fare"]
cat_cols=["Pclass","Sex","Cabin","Embarked","demograf"]
bin_cols=["Age","SibSp","Parch"]
num_si_step=("num_si",SimpleImputer(strategy='mean'))
cat_si_step=("cat_si",SimpleImputer(strategy="constant",fill_value="Missing"))
bin_si_step=("bin_si",SimpleImputer(strategy="median"))
num_2_step=("num_2",StandardScaler())
cat_2_step=("cat_2",OneHotEncoder(sparse=False, handle_unknown='ignore'))
bin_2_step=("bin_2",KBinsDiscretizer(encode='onehot-dense',n_bins=5))
num_steps=[num_si_step,num_2_step]
cat_steps=[cat_si_step,cat_2_step]
bin_steps=[bin_si_step,bin_2_step]
num_pipe=Pipeline(num_steps)
cat_pipe=Pipeline(cat_steps)
bin_pipe=Pipeline(bin_steps)
para_trans=[('cat', cat_pipe, cat_cols),('num',num_pipe,num_cols),('bin',bin_pipe,bin_cols)]
trans=ColumnTransformer(transformers=para_trans)
X_modif=trans.fit_transform(X)
print(X_modif.shape)
print(X.groupby("Cabin")["Cabin"].nunique())
print(X.groupby("Cabin")["Cabin"].nunique().count())

"""La primer gran decisión a tomar en términos de feature engeenering es que hacer con la variable
Cabin. Sacarla o dejarla? Si la dejamos, la dejamos tal cual está sin pérdida de información pero
con el riesgo de overfittear los modelos? O la seaparamos con o sin información? O tal vez podamos
encontrar una estrategia para armar diferentes grupos de cabinas"""

X_sin=X.copy()
X_sin.loc[:,"Cabin"]="Constant" #Dataset quitando la variable Cabin
X_sin=X_sin.iloc[0:891,:]
print(X.head(10))
print(X_sin.head(10))
print(X_sin.iloc[0:1,:])
print(X.iloc[0:1,:])
X_dif=X.copy()
X_dif.loc[~pd.isnull(X_dif["Cabin"]),"Cabin"]="Cabina"
X_dif.loc[pd.isnull(X_dif["Cabin"]),"Cabin"]="Sin Cabina" #Dataset separando en con y sin cabina
print(X_dif["Cabin"].describe())
print(X_dif.tail(40))
X_dif_modif=trans.fit_transform(X_dif)
print(X_dif_modif.shape)


""" Luego de haber realizado el preprocesamiento de los datos obtuvimos 3 datasets para aplicarles los
algoritmos. Estos 3 datasets solamente difieren en la variable "Cabin": tenemos allí un dataset que contiene
la variable tal cual, un segundo dataset que eliminó la variable y un 3er dataset que la transformó en una
variable binaria de si hay información de la cabina o no. A su vez, aplicaremos sobre este dataset 7 algoritmos
de ML diferentes: 1) Regresión lógistica 2) LinearDiscriminantAnalysis 3) Naive Bayes 4) Support Vector Machine
5) Árbol de decisión 6) Random Forest 7) Red Neuronal. Es decir que realizaremos 3x7=21 modelos. A su vez, para
varios de estos modelos se realizará la optimización de hiperparámetros. La estrategia a seguir será la
siguiente: se aplicará cada algoritmo a los 3 datasets propuestos con 30 semillas diferentes para separar los
datasets en train y test. De allí se determinará para cada algoritmo cual es el dataset con el que se consiguen
resultados más precisos. Para cada algoritmo se elegirá 1 modelo (aplicación sobre 1 dataset) finalista. Por
último se compararán los 7 datasets finalistas determinando cual es el mejor para predecir cuales fueron los
pasajeros sobrevivientes en el titanic."""


#1) Regresión Logística + 2) Linear Discriminant Analysis + 3) Naive Bayes

regresion_con=[]
regresion_sin=[]
regresion_bin=[]
regre_con=LogisticRegression()
regre_sin=LogisticRegression()
regre_bin=LogisticRegression()

linear_con=[]
linear_sin=[]
linear_bin=[]
lin_con=LinearDiscriminantAnalysis()
lin_sin=LinearDiscriminantAnalysis()
lin_bin=LinearDiscriminantAnalysis()

naive_con=[]
naive_sin=[]
naive_bin=[]
nb_con=GaussianNB()
nb_sin=GaussianNB()
nb_bin=GaussianNB()

for i in semillas:
    X_train_con, X_test_con, Y_train_con, Y_test_con=train_test_split(X,Y,random_state=i)
    X_train_sin, X_test_sin, Y_train_sin, Y_test_sin=train_test_split(X_sin,Y,random_state=i)
    X_train_bin, X_test_bin, Y_train_bin, Y_test_bin=train_test_split(X_dif,Y,random_state=i)

    X_train_con_trans=trans.fit_transform(X_train_con)
    X_test_con_trans=trans.transform(X_test_con)
    X_train_sin_trans=trans.fit_transform(X_train_sin)
    X_test_sin_trans=trans.transform(X_test_sin)
    X_train_bin_trans=trans.fit_transform(X_train_bin)
    X_test_bin_trans=trans.transform(X_test_bin)

    regre_con.fit(X_train_con_trans,Y_train_con)
    regresion_con.append(accuracy_score(regre_con.predict(X_test_con_trans),Y_test_con))
    regre_sin.fit(X_train_sin_trans,Y_train_sin)
    regresion_sin.append(accuracy_score(regre_sin.predict(X_test_sin_trans),Y_test_sin))
    regre_bin.fit(X_train_bin_trans,Y_train_bin)
    regresion_bin.append(accuracy_score(regre_bin.predict(X_test_bin_trans),Y_test_bin))

    lin_con.fit(X_train_con_trans,Y_train_con)
    linear_con.append(accuracy_score(lin_con.predict(X_test_con_trans),Y_test_con))
    lin_sin.fit(X_train_sin_trans,Y_train_sin)
    linear_sin.append(accuracy_score(lin_sin.predict(X_test_sin_trans),Y_test_sin))
    lin_bin.fit(X_train_bin_trans,Y_train_bin)
    linear_bin.append(accuracy_score(lin_bin.predict(X_test_bin_trans),Y_test_bin))

    nb_con.fit(X_train_con_trans,Y_train_con)
    naive_con.append(accuracy_score(nb_con.predict(X_test_con_trans),Y_test_con))
    nb_sin.fit(X_train_sin_trans,Y_train_sin)
    naive_sin.append(accuracy_score(nb_sin.predict(X_test_sin_trans),Y_test_sin))
    nb_bin.fit(X_train_bin_trans,Y_train_bin)
    naive_bin.append(accuracy_score(nb_bin.predict(X_test_bin_trans),Y_test_bin))

plt.subplot(311)
plt.hist(regresion_con,5,color="blue",label="Con variable Cabina",density=1)    
plt.subplot(312)
plt.hist(regresion_sin,5,color="red",label="Sin variable Cabina",density=1)
plt.subplot(313)
plt.hist(regresion_bin,5,color="green",label="Variable Cabina binaria",density=1)
plt.suptitle("Regresión Logística")
plt.show()
plt.plot(range(30),regresion_con,'b-',label="Con variable Cabina")
plt.plot(range(30),regresion_sin,'r-',label="Sin variable Cabina")
plt.plot(range(30),regresion_bin,'g-',label="Variable Cabina binaria")
plt.legend(loc="best")
plt.title("Regresión Logística",size=20)
plt.ylabel("Accuracy Score",size=12)
plt.show()
plt.plot(range(30),np.sort(regresion_con),'b-',label="Con variable Cabina")
plt.plot(range(30),np.sort(regresion_sin),'r-',label="Sin variable Cabina")
plt.plot(range(30),np.sort(regresion_bin),'g-',label="Variable Cabina binaria")
plt.legend(loc="best")
plt.title("Regresión Logística",size=20)
plt.ylabel("Accuracy Score",size=12)
plt.show()
#Para la reg logística elegímos el dataset con variable cabina binaria

plt.subplot(311)
plt.hist(linear_con,5,color="blue",label="Con variable Cabina",density=1)    
plt.subplot(312)
plt.hist(linear_sin,5,color="red",label="Sin variable Cabina",density=1)
plt.subplot(313)
plt.hist(linear_bin,5,color="green",label="Variable Cabina binaria",density=1)
plt.suptitle("Linear Discriminant Analysis")
plt.show()
plt.plot(range(30),linear_con,'b-',label="Con variable Cabina")
plt.plot(range(30), linear_sin,'r-',label="Sin variable Cabina")
plt.plot(range(30),linear_bin,'g-',label="Variable Cabina binaria")
plt.legend(loc="best")
plt.title("Linear Discriminant Analysis",size=20)
plt.ylabel("Accuracy Score",size=12)
plt.show()
plt.plot(range(30),np.sort(linear_con),'b-',label="Con variable Cabina")
plt.plot(range(30), np.sort(linear_sin),'r-',label="Sin variable Cabina")
plt.plot(range(30),np.sort(linear_bin),'g-',label="Variable Cabina binaria")
plt.legend(loc="best")
plt.title("Linear Discriminant Analysis",size=20)
plt.ylabel("Accuracy Score",size=12)
plt.show()
#Para el discriminant analysis elegimos el dataset CON la variable cabina

plt.subplot(311)
plt.hist(naive_con,5,color="blue",label="Con variable Cabina",density=1)    
plt.subplot(312)
plt.hist(naive_sin,5,color="red",label="Sin variable Cabina",density=1)
plt.subplot(313)
plt.hist(naive_bin,5,color="green",label="Variable Cabina binaria",density=1)
plt.suptitle("Naive Bayes")
plt.show()
plt.plot(range(30),naive_con,'b-',label="Con variable Cabina")
plt.plot(range(30), naive_sin,'r-',label="Sin variable Cabina")
plt.plot(range(30),naive_bin,'g-',label="Variable Cabina binaria")
plt.legend(loc="best")
plt.title("Naive Bayes",size=20)
plt.ylabel("Accuracy Score",size=12)
plt.show()
plt.plot(range(30),np.sort(naive_con),'b-',label="Con variable Cabina")
plt.plot(range(30), np.sort(naive_sin),'r-',label="Sin variable Cabina")
plt.plot(range(30),np.sort(naive_bin),'g-',label="Variable Cabina binaria")
plt.legend(loc="best")
plt.title("Naive Bayes",size=20)
plt.ylabel("Accuracy Score",size=12)
plt.show()
#Para Naive Bayes elegímos el dataset con variable cabina binaria

#4) Árbol de deicsión

promedio_con=0
promedio_sin=0
promedio_bin=0
profundidad_con=0
profundidad_sin=0
profundidad_bin=0
hoja_con=0
hoja_sin=0
hoja_bin=0
split_con=0
split_sin=0
split_bin=0
arbolito_con=[]
arbolito_sin=[]
arbolito_bin=[]
arbolito_con_2=[]
arbolito_sin_2=[]
arbolito_bin_2=[]

for i in range(1,8,1):
    for j in range(3,6,1):
        for k in range(4,8,1):
            for l in semillas:

                X_train_con, X_test_con, Y_train_con, Y_test_con=train_test_split(X,Y,random_state=l)
                X_train_sin, X_test_sin, Y_train_sin, Y_test_sin=train_test_split(X_sin,Y,random_state=l)
                X_train_bin, X_test_bin, Y_train_bin, Y_test_bin=train_test_split(X_dif,Y,random_state=l)

                X_train_con_trans=trans.fit_transform(X_train_con)
                X_test_con_trans=trans.transform(X_test_con)
                X_train_sin_trans=trans.fit_transform(X_train_sin)
                X_test_sin_trans=trans.transform(X_test_sin)
                X_train_bin_trans=trans.fit_transform(X_train_bin)
                X_test_bin_trans=trans.transform(X_test_bin)

                arbol_con=DecisionTreeClassifier(max_depth=i,min_samples_leaf=j,min_samples_split=k)
                arbol_sin=DecisionTreeClassifier(max_depth=i,min_samples_leaf=j,min_samples_split=k)
                arbol_bin=DecisionTreeClassifier(max_depth=i,min_samples_leaf=j,min_samples_split=k)

                arbol_con.fit(X_train_con_trans,Y_train_con)
                arbol_sin.fit(X_train_sin_trans,Y_train_sin)
                arbol_bin.fit(X_train_bin_trans,Y_train_bin)
                arbolito_con.append(accuracy_score(arbol_con.predict(X_test_con_trans),Y_test_con))
                arbolito_sin.append(accuracy_score(arbol_sin.predict(X_test_sin_trans),Y_test_sin))
                arbolito_bin.append(accuracy_score(arbol_bin.predict(X_test_bin_trans),Y_test_bin))

            if np.mean(np.asarray(arbolito_con))>promedio_con:
                promedio_con=np.mean(np.asarray(arbolito_con))
                arbolito_con_2.clear()
                arbolito_con_2=arbolito_con.copy()
                profundidad_con=i
                hoja_con=j
                split_con=k
            arbolito_con.clear()

            if np.mean(arbolito_sin)>promedio_sin:
                promedio_sin=np.mean(arbolito_sin)
                arbolito_sin_2.clear()
                arbolito_sin_2=arbolito_sin.copy()
                profundidad_sin=i
                hoja_sin=j
                split_sin=k
            arbolito_sin.clear()

            if np.mean(arbolito_bin)>promedio_bin:
                promedio_bin=np.mean(arbolito_bin)
                arbolito_bin_2.clear()
                arbolito_bin_2=arbolito_bin.copy()
                profundidad_bin=i
                hoja_bin=j
                split_bin=k
            arbolito_bin.clear()

print("Mejores parámetros del Árbol con:\n")
print("Max depth: {}".format(profundidad_con))
print("Min leaf: {}".format(hoja_con))
print("Min split: {}".format(split_con))
print("Accuracy scores obtenidos:\n {}".format(arbolito_con_2))
print("Mejores parámetros del Árbol sin:\n")
print("Max depth: {}".format(profundidad_sin))
print("Min leaf: {}".format(hoja_sin))
print("Min split: {}".format(split_sin))
print("Accuracy scores obtenidos:\n {}".format(arbolito_sin_2))
print("Mejores parámetros del Árbol bin:\n")
print("Max depth: {}".format(profundidad_bin))
print("Min leaf: {}".format(hoja_bin))
print("Min split: {}".format(split_bin))
print("Accuracy scores obtenidos:\n {}".format(arbolito_bin_2))

plt.subplot(311)
plt.hist(arbolito_con_2,5,color="blue",label="Con variable Cabina",density=1)    
plt.subplot(312)
plt.hist(arbolito_sin_2,5,color="red",label="Sin variable Cabina",density=1)
plt.subplot(313)
plt.hist(arbolito_bin_2,5,color="green",label="Variable Cabina binaria",density=1)
plt.suptitle("Árbol de decisión")
plt.show()
plt.plot(range(30),arbolito_con_2,'b-',label="Con variable Cabina")
plt.plot(range(30),arbolito_sin_2,'r-',label="Sin variable Cabina")
plt.plot(range(30),arbolito_bin_2,'g-',label="Variable Cabina binaria")
plt.legend(loc="best")
plt.title("Árbol de decisión",size=20)
plt.ylabel("Accuracy Score",size=12)
plt.show()
plt.plot(range(30),np.sort(arbolito_con_2),'b-',label="Con variable Cabina")
plt.plot(range(30), np.sort(arbolito_sin_2),'r-',label="Sin variable Cabina")
plt.plot(range(30),np.sort(arbolito_bin_2),'g-',label="Variable Cabina binaria")
plt.legend(loc="best")
plt.title("Árbol de decisión",size=20)
plt.ylabel("Accuracy Score",size=12)
plt.show()
#Para el árbol de decisión elegímos el dataset con variable cabina binaria

#5) Random Forest 

promedio_con=0
promedio_sin=0
promedio_bin=0
profundidad_con=0
profundidad_sin=0
profundidad_bin=0
hoja_con=0
hoja_sin=0
hoja_bin=0
split_con=0
split_sin=0
split_bin=0
estimators_con=0
estimators_sin=0
estimators_bin=0
bosquecito_con=[]
bosquecito_sin=[]
bosquecito_bin=[]
bosquecito_con_2=[]
bosquecito_sin_2=[]
bosquecito_bin_2=[]

for i in range(1,8,1):
    for j in range(3,6,1):
        for k in range(4,8,1):
            for m in range(20,100,20):
                for l in semillas:

                    X_train_con, X_test_con, Y_train_con, Y_test_con=train_test_split(X,Y,random_state=l)
                    X_train_sin, X_test_sin, Y_train_sin, Y_test_sin=train_test_split(X_sin,Y,random_state=l)
                    X_train_bin, X_test_bin, Y_train_bin, Y_test_bin=train_test_split(X_dif,Y,random_state=l)

                    X_train_con_trans=trans.fit_transform(X_train_con)
                    X_test_con_trans=trans.transform(X_test_con)
                    X_train_sin_trans=trans.fit_transform(X_train_sin)
                    X_test_sin_trans=trans.transform(X_test_sin)
                    X_train_bin_trans=trans.fit_transform(X_train_bin)
                    X_test_bin_trans=trans.transform(X_test_bin)

                    bosque_con=RandomForestClassifier(max_depth=i,min_samples_leaf=j,min_samples_split=k,n_estimators=m)
                    bosque_sin=RandomForestClassifier(max_depth=i,min_samples_leaf=j,min_samples_split=k,n_estimators=m)
                    bosque_bin=RandomForestClassifier(max_depth=i,min_samples_leaf=j,min_samples_split=k,n_estimators=m)

                    bosque_con.fit(X_train_con_trans,Y_train_con)
                    bosque_sin.fit(X_train_sin_trans,Y_train_sin)
                    bosque_bin.fit(X_train_bin_trans,Y_train_bin)
                    bosquecito_con.append(accuracy_score(bosque_con.predict(X_test_con_trans),Y_test_con))
                    bosquecito_sin.append(accuracy_score(bosque_sin.predict(X_test_sin_trans),Y_test_sin))
                    bosquecito_bin.append(accuracy_score(bosque_bin.predict(X_test_bin_trans),Y_test_bin))

                if np.mean(np.asarray(bosquecito_con))>promedio_con:
                    promedio_con=np.mean(np.asarray(bosquecito_con))
                    bosquecito_con_2.clear()
                    bosquecito_con_2=bosquecito_con.copy()
                    profundidad_con=i
                    hoja_con=j
                    split_con=k
                    estimators_con=m
                bosquecito_con.clear()

                if np.mean(bosquecito_sin)>promedio_sin:
                    promedio_sin=np.mean(bosquecito_sin)
                    bosquecito_sin_2.clear()
                    bosquecito_sin_2=bosquecito_sin.copy()
                    profundidad_sin=i
                    hoja_sin=j
                    split_sin=k
                    estimators_sin=m
                bosquecito_sin.clear()

                if np.mean(bosquecito_bin)>promedio_bin:
                    promedio_bin=np.mean(bosquecito_bin)
                    bosquecito_bin_2.clear()
                    bosquecito_bin_2=bosquecito_bin.copy()
                    profundidad_bin=i
                    hoja_bin=j
                    split_bin=k
                    estimators_bin=m
                bosquecito_bin.clear()

print("Mejores parámetros del Random Forest con:\n")
print("Max depth: {}".format(profundidad_con))
print("Min leaf: {}".format(hoja_con))
print("Min split: {}".format(split_con))
print("Número de bosques: {}".format(estimators_con))
print("Accuracy scores obtenidos:\n {}".format(bosquecito_con_2))

print("Mejores parámetros del Random Forest sin:\n")
print("Max depth: {}".format(profundidad_sin))
print("Min leaf: {}".format(hoja_sin))
print("Min split: {}".format(split_sin))
print("Número de bosques: {}".format(estimators_sin))
print("Accuracy scores obtenidos:\n {}".format(bosquecito_sin_2))

print("Mejores parámetros del Random Forest bin:\n")
print("Max depth: {}".format(profundidad_bin))
print("Min leaf: {}".format(hoja_bin))
print("Min split: {}".format(split_bin))
print("Número de bosques: {}".format(estimators_bin))
print("Accuracy scores obtenidos:\n {}".format(arbolito_bin_2))

plt.subplot(311)
plt.hist(bosquecito_con_2,5,color="blue",label="Con variable Cabina",density=1)    
plt.subplot(312)
plt.hist(bosquecito_sin_2,5,color="red",label="Sin variable Cabina",density=1)
plt.subplot(313)
plt.hist(bosquecito_bin_2,5,color="green",label="Variable Cabina binaria",density=1)
plt.suptitle("Random Forest")
plt.show()
plt.plot(range(30),bosquecito_con_2,'b-',label="Con variable Cabina")
plt.plot(range(30),bosquecito_sin_2,'r-',label="Sin variable Cabina")
plt.plot(range(30),bosquecito_bin_2,'g-',label="Variable Cabina binaria")
plt.legend(loc="best")
plt.title("Random Forest",size=20)
plt.ylabel("Accuracy Score",size=12)
plt.show()
plt.plot(range(30),np.sort(bosquecito_con_2),'b-',label="Con variable Cabina")
plt.plot(range(30), np.sort(bosquecito_sin_2),'r-',label="Sin variable Cabina")
plt.plot(range(30),np.sort(bosquecito_bin_2),'g-',label="Variable Cabina binaria")
plt.legend(loc="best")
plt.title("Random Forest",size=20)
plt.ylabel("Accuracy Score",size=12)
plt.show()

# 6) Red neuronal

promedio_con=0
promedio_sin=0
promedio_bin=0
neuronas_con=0
neuronas_sin=0
neuronas_bin=0
alfa_con=0
alfa_sin=0
alfa_bin=0
red_con=[]
red_sin=[]
red_bin=[]
red_con_2=[]
red_sin_2=[]
red_bin_2=[]

for a in range(1,20,1):
    for b in np.logspace(-4,3,8):
        for c in semillas:

            X_train_con, X_test_con, Y_train_con, Y_test_con=train_test_split(X,Y,random_state=c)
            X_train_sin, X_test_sin, Y_train_sin, Y_test_sin=train_test_split(X_sin,Y,random_state=c)
            X_train_bin, X_test_bin, Y_train_bin, Y_test_bin=train_test_split(X_dif,Y,random_state=c)

            X_train_con_trans=trans.fit_transform(X_train_con)
            X_test_con_trans=trans.transform(X_test_con)
            X_train_sin_trans=trans.fit_transform(X_train_sin)
            X_test_sin_trans=trans.transform(X_test_sin)
            X_train_bin_trans=trans.fit_transform(X_train_bin)
            X_test_bin_trans=trans.transform(X_test_bin)

            nn_con=MLPClassifier(hidden_layer_sizes=(a,),alpha=b)
            nn_sin=MLPClassifier(hidden_layer_sizes=(a,),alpha=b)
            nn_bin=MLPClassifier(hidden_layer_sizes=(a,),alpha=b)

            nn_con.fit(X_train_con_trans,Y_train_con)
            nn_sin.fit(X_train_sin_trans,Y_train_sin)
            nn_bin.fit(X_train_bin_trans,Y_train_bin)
            red_con.append(accuracy_score(nn_con.predict(X_test_con_trans),Y_test_con))
            red_sin.append(accuracy_score(nn_sin.predict(X_test_sin_trans),Y_test_sin))
            red_bin.append(accuracy_score(nn_bin.predict(X_test_bin_trans),Y_test_bin))

        if np.mean(np.asarray(red_con))>promedio_con:
            promedio_con=np.mean(np.asarray(red_con))
            red_con_2.clear()
            red_con_2=red_con.copy()
            neuronas_con=a
            alfa_con=b
        red_con.clear()

        if np.mean(np.asarray(red_sin))>promedio_sin:
            promedio_sin=np.mean(np.asarray(red_sin))
            red_sin_2.clear()
            red_sin_2=red_sin.copy()
            neuronas_sin=a
            alfa_sin=b
        red_sin.clear()

        if np.mean(np.asarray(red_bin))>promedio_bin:
            promedio_bin=np.mean(np.asarray(red_bin))
            red_bin_2.clear()
            red_bin_2=red_bin.copy()
            neuronas_bin=a
            alfa_bin=b
        red_bin.clear()

print("Mejores parámetros de la red neuronal con:\n")
print("Cantidad de neuronas de la capa oculta: {}".format(neuronas_con))
print("Alfa del mejor modelo:{}".format(alfa_con))
print("Accuracy scores obtenidos \n {}".format(red_con_2))
print("Mejores parámetros de la red neuronal sin:\n")
print("Cantidad de neuronas de la capa oculta: {}".format(neuronas_sin))
print("Alfa del mejor modelo:{}".format(alfa_sin))
print("Accuracy scores obtenidos \n {}".format(red_sin_2))
print("Mejores parámetros de la red neuronal bin:\n")
print("Cantidad de neuronas de la capa oculta: {}".format(neuronas_bin))
print("Alfa del mejor modelo:{}".format(alfa_bin))
print("Accuracy scores obtenidos \n {}".format(red_bin_2))

plt.subplot(311)
plt.hist(red_con_2,5,color="blue",label="Con variable Cabina",density=1)    
plt.subplot(312)
plt.hist(red_sin_2,5,color="red",label="Sin variable Cabina",density=1)
plt.subplot(313)
plt.hist(red_bin_2,5,color="green",label="Variable Cabina binaria",density=1)
plt.suptitle("Red Neuronal")
plt.show()
plt.plot(range(30),red_con_2,'b-',label="Con variable Cabina")
plt.plot(range(30),red_sin_2,'r-',label="Sin variable Cabina")
plt.plot(range(30),red_bin_2,'g-',label="Variable Cabina binaria")
plt.legend(loc="best")
plt.title("Red Neuronal",size=20)
plt.ylabel("Accuracy Score",size=12)
plt.show()
plt.plot(range(30),np.sort(red_con_2),'b-',label="Con variable Cabina")
plt.plot(range(30), np.sort(red_sin_2),'r-',label="Sin variable Cabina")
plt.plot(range(30),np.sort(red_bin_2),'g-',label="Variable Cabina binaria")
plt.legend(loc="best")
plt.title("Red Neuronal",size=20)
plt.ylabel("Accuracy Score",size=12)
plt.show()


"""
Luego de corridos todos los modelos con las 3 variantes de datasets, obtuvimos los 6 algoritmos finalistas
que han de compararse entre sí. Estos algoritmos finalistas son:
1) Regresión logística con variable cabina binaria
2) Linear Discriminant Analysis con variable cabina normal
3) Naive Bayes con variable cabina binaria
4) Árbol de decisión con variable cabina binaria
5) Random Forest sin variable cabina
6) Red neuronal con variable cabina binaria
Vamos a colocar en un gráfico de forma conjunta los resultados obtenidos al inicar el modelo con diferentes
semillas y gráfiamente determinar cual sería el mejor modelo para realizar las predicciones
"""

plt.plot(range(30),np.sort(regresion_bin),label="Regresión Logística")
plt.plot(range(30),np.sort(linear_con),label="Linear Discriminant Analysis")
plt.plot(range(30),np.sort(naive_bin),label="Naive Bayes")
plt.plot(range(30),np.sort(arbolito_bin_2),label="Árbol de decisión")
plt.plot(range(30),np.sort(bosquecito_sin_2),label="Random Forest")
plt.plot(range(30),np.sort(red_bin_2),label="Red neuronal")
plt.ylabel("Accuracy score")
plt.title("Comparativa de modelos",size=20)
plt.legend(loc="best")
plt.show()


"""
Gráficamente se observa que el modelo que mejor predecir los datos es la red neuronal que utiliza
el dataset con la variable cabina binaria. Para finalizar este programa, se procede a evaluar los
parámetros de este modeo y utilizarlos en el dataset de testeo real para poder predecir los restantes
sobrevivientes al hundimiento del titanic.
"""

mejor_algoritmo=MLPClassifier(hidden_layer_sizes=(neuronas_bin,),alpha=alfa_bin)
random.seed=0
X_train, X_test, Y_train, Y_test= train_test_split(X_dif,Y,random_state=0)
X_train_final=trans.fit_transform(X_train)
X_test_final=trans.transform(X_test)
mejor_algoritmo.fit(X_train_final,Y_train)
sobrevivientes=mejor_algoritmo.predict(X_test_final)
print("\n Parámetros del mejor algoritmo:\n")
print("Accuracy score: {}".format(accuracy_score(sobrevivientes,Y_test)))
print("Matriz de confusión: {}".format(confusion_matrix(sobrevivientes,Y_test)))

"""
Por último, utilizaremos nuestro mejor algoritmo para predecir los datos no rotulados. Esta vez
entrenaremos el algoritmo con todo el dataset X.
"""

mejor_algoritmo_2=MLPClassifier(hidden_layer_sizes=(neuronas_bin,),alpha=alfa_bin)
titanic_2=pd.read_csv("data/test.csv",sep=',')
Z=titanic_2.iloc[:,1:]
Nombre=Z.pop("Name")
Z.pop("Ticket")
Z["demograf"]="NA"
Z.loc[(Z["Age"]<=18) & (Z["Sex"]=="female"),"demograf"]="Niña"
Z.loc[(Z["Age"]<=18) & (Z["Sex"]=="male"),"demograf"]="Niño"
Z.loc[(Z["Age"]>=65) & (Z["Sex"]=="female"),"demograf"]="Anciana"
Z.loc[(Z["Age"]>=65) & (Z["Sex"]=="male"),"demograf"]="Anciano"
Z.loc[(Z["Age"]<65) & (Z["Age"]>18) & (Z["Sex"]=="female"),"demograf"]="Adulta"
Z.loc[(Z["Age"]<65) & (Z["Age"]>18) & (Z["Sex"]=="male"),"demograf"]="Adulto"
Z_dif=Z.copy()
Z_dif.loc[~pd.isnull(Z_dif["Cabin"]),"Cabin"]="Cabina"
Z_dif.loc[pd.isnull(Z_dif["Cabin"]),"Cabin"]="Sin Cabina"
X_dif_final=trans.fit_transform(X_dif)
Z_dif_final=trans.transform(Z_dif)
mejor_algoritmo_2.fit(X_dif_final,Y)
print("Predicción de sobrevivientes:{}".format(np.concatenate(Nombre,mejor_algoritmo_2.predict(Z_dif_final),axis=1)))
