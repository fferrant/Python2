"""
En este programa se intentará predecir el salario mensual de empleados de una firma a partir de
algoritmos que utilicen tanto datos numéricos (algunos de los cuales se agruparán en bines) y
datos categóricos. Se utilizará a tal fin una red neuronal y un random forest. Para el tuneo de
hiperparámetros, se recurrira a la herramienta de grid search.
"""


#Carga de librerías a utilizar
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import time
import random
import pandas as pd 
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold


#Cargamos el dataset y realizamos un análisis exploratorio del mismo
X=pd.read_csv("data/employees.csv",sep=';')
print(X.head(10))
print("Forma del dataset: {}".format(X.shape))
print("Columnas del dataset:{}".format(X.columns))
X=X.dropna()
print("Forma del dataset: {}".format(X.shape))
Y=X.pop("MonthlyIncome").values
X=X.drop(['EmployeeCount','EmployeeNumber'],axis=1)
print("Tipo de datos:{}".format(X.dtypes))


#Obtenemos nuestros strings de columnas previo al preprocesamiento de datos
kinds = np.array([dt.kind for dt in X.dtypes])
print(kinds)
columns=X.columns.values
is_cat=kinds=='O'
print(is_cat)
cat_cols=columns[is_cat] #Lista de columnas que son strings
print(cat_cols)
num_cols=['DailyRate','DistanceFromHome','HourlyRate','MonthlyRate','PercentSalaryHike',
          'StandardHours','TotalWorkingYears'] #Lista de columnas numércias
bin_cols=columns[~is_cat]
bin_cols=list(bin_cols)
for i in num_cols:
    bin_cols.remove(i)
print(bin_cols)


#Ahora si pasamos al preprocesamiento del dataset

si_cat_step=('si1',SimpleImputer(strategy='constant',fill_value='MISSING'))
ohe_cat_step=('ohe',OneHotEncoder(sparse=False,handle_unknown='ignore'))
cat_steps=[si_cat_step,ohe_cat_step]
cat_pipe=Pipeline(cat_steps)

si_num_step=('si2',SimpleImputer(strategy='mean'))
ss_num_step=('ss',StandardScaler())
num_steps=[si_num_step,ss_num_step]
num_pipe=Pipeline(num_steps)

si_bin_step=('si3',SimpleImputer(strategy='median'))
kb_bin_step=('kb',KBinsDiscretizer(encode='onehot-dense'))
bin_steps=[si_bin_step,kb_bin_step]
bin_pipe=Pipeline(bin_steps)

transformers=[('cat',cat_pipe,cat_cols),('num',num_pipe,num_cols),('bin',bin_pipe,bin_cols)]
ct=ColumnTransformer(transformers=transformers)
Z=ct.fit_transform(X)
print(Z.shape)


"""
Ahora que tenemos nuestro dataset pre-procesado realizaremos 2 algoritmos de ML para nuestra
regresión: 1) Red Neuronal 2) Random Forest. En ambos casos el tuneo de hiperparámetros se
realizará a través de las herramientas de Grid Search y Cross Validation. Sobre los modelos
con mejores parámetros se tomará la métrica de R2 para determinar que modelo predice mejor
el salario de los empleados.
"""


# 1) Red neuronal

kf=KFold(n_splits=5, shuffle=True, random_state=123)
parametros_red = {
    "hidden_layer_sizes":range(1,31,3),
    "alpha":np.logspace(-5,2,8)
}
modelo_red=MLPRegressor(solver='lbfgs',random_state=123)
grid_red=GridSearchCV(modelo_red,parametros_red,cv=kf)
grid_red.fit(Z,Y)
print("Mejores parámetros del modelo:{}".format(grid_red.best_params_))
print("Score del mejor modelo:{}".format(grid_red.best_score_))
print("Resultados obtenidos durante el grid search:\n{}".format(grid_red.cv_results_))


# 2) Random forest

parametros_bosque = {
    "max_depth": range(1, 7, 2),
    "min_samples_split": range(2, 18, 4),
    "min_samples_leaf":range(1,9,2),
    "n_estimators":range(10,30,10)
}
modelo_bosque=RandomForestRegressor(random_state=123)
grid_bosque=GridSearchCV(modelo_bosque,parametros_bosque,cv=kf)
grid_bosque.fit(Z,Y)
print("Mejores parámetros del modelo:{}".format(grid_bosque.best_params_))
print("Score del mejor modelo:{}".format(grid_bosque.best_score_))
print("Resultados obtenidos durante el grid search:\n{}".format(grid_bosque.cv_results_))

#Generamos modelos con los mejores parámetros de cada algoritmo

semillas=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
red_scores=[]
bosque_scores=[]
la_red=MLPRegressor(alpha=0.00001, hidden_layer_sizes= (4,),solver='lbfgs')
el_bosque=RandomForestRegressor(max_depth= 5, min_samples_leaf=1, min_samples_split=14, n_estimators=20)
for i in semillas:
    Z_train, Z_test, Y_train, Y_test = train_test_split(Z,Y,random_state=i)
    la_red.fit(Z_train,Y_train)
    red_scores.append(r2_score(la_red.predict(Z_test),Y_test))
    el_bosque.fit(Z_train,Y_train)
    bosque_scores.append(r2_score(el_bosque.predict(Z_test),Y_test))

print("Scores de la mejor red:\n{}".format(red_scores))
print("Scores del mejor bosque:\n{}".format(bosque_scores))
np_red=np.asarray(red_scores)
np_bosque=np.asarray(bosque_scores)
print("Promedio de scores de la mejor red:\n{}".format(np_red.mean()))
print("Promedio de scores del mejor bosque:\n{}".format(np_bosque.mean()))

sns.boxplot(y=red_scores,color='green')
plt.title("Distribución de scores de la mejor red",size=18)
plt.show()
sns.boxplot(y=bosque_scores,color='green')
plt.title("Distribución de scores del mejor bosque",size=18)
plt.show()


"""
Luego de entrenados y testeados los 2 algoritmos propuestos, concluimos que el algoritmo de random
forest predice mejor los sueldos de empleados que la red neuronal, obteniendo un R2 promedio de 0.93
vs 0.91 obtenido por la red neuronal.
"""
