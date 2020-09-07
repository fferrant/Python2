
"""
Utilizaremos un dataset de Kaggle para realizar un análisis exploratorio de series de tiempo con datos ambientales.
El dataset consta de las mediciones tomadas por 3 sensores en diferentes lugares geográficos entre el 
12/07/2020 00:00:00 UTC y el 19/07/2020 23:59:59 UTC. Las columnas que encontramos en este dataset son las siguientes:

device: identificador único de cada sensor
co: niveles monóxido de carbono
humidity: humedad
lpg: niveles de gas licuado del petróleo en aire
smoke: niveles d ehumo en el aire
temp: temperatura
motion: sensado de movimiento
light: detector lumínico

"""

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


#Cargamos el dataset y obtenemos algunos valores del mismo
sensores=pd.read_csv('data/telemetry.csv')
print("\n Primeros 5 valores del dataset:\n")
print(sensores.head(5))
print("\n Últimos 5 valores del dataset:\n")
print(sensores.tail(5))
print("\n Descripción de cada columna:\n")
print(sensores.describe())
print("\n Nombre de las columnas:\n")
print(sensores.columns)
print("\n Dimensiones del dataset:\n")
print(sensores.shape)
print("\n Cantidad de sensores:\n")
print(sensores["device"].nunique())
print("\n Cantidad de datos tomados por cada sensor:\n")
print(sensores.groupby("device").count())
agrupados=sensores.groupby("device")
print("\n Parámetros de tiempo agrupados para cada sensor:\n")
print(agrupados["ts"].describe()) #Con esto podemos observar si los 3 sensores cubren el mismo espacio temporal
print("\n Inicio temporal de cada sensor:\n")
print(agrupados["ts"].min()) #Nos interesa si tanto el valor máximo como el mínimo son iguales o muy similares
agrupados2=sensores.groupby(["light","motion"])
print(agrupados2.describe()) #Verificamos como se comportan los datos frente a datos categóricos


#Análisis de valores nulos en nuestra información
msno.matrix(sensores)
plt.show()
msno.bar(sensores)
plt.show()
msno.heatmap(sensores)
plt.show()
de_nulos=sensores[pd.isnull(sensores).any(axis=1)]
print("\n Cantidad de filas con algún valor nulo:\n")
print(de_nulos["ts"].count()) #A partir de métodos diferentes llegamos a la conclusión de que no hay NANs en los datos


#Realizamos diferentes gráficos para comparar los datos tomados por cada uno de los sensores
plt.figure(figsize=(50,15))
plt.subplot()
for i in sensores["device"].unique():
    plt.plot(sensores[sensores["device"]==i]["ts"],sensores[sensores["device"]==i]["temp"],label=i) #Análisis de temperaturas
plt.show() #Este gráfico es inmirable!!! Busquemos alternativas para nuestro gráfico


#Alternativa 2: Partimos nuestro dataset en 3 datasets ordenados temporalmente
partidor=int(len(sensores)/5)
particion1=sensores.iloc[0:partidor-1,]
particion2=sensores.iloc[partidor:2*partidor-1,]
particion3=sensores.iloc[2*partidor:3*partidor-1,]
particion4=sensores.iloc[3*partidor:4*partidor-1,]
particion5=sensores.iloc[4*partidor:5*partidor-3,]
def graf(par):
    for i in par["device"].unique():
        plt.plot(par[par["device"]==i]["ts"],par[par["device"]==i]["temp"],label=i)
plt.figure(figsize=(50,30))
plt.subplot(511)
graf(particion5)
plt.subplot(512)
graf(particion4)
plt.subplot(513)
graf(particion3)
plt.subplot(514)
graf(particion2)
plt.subplot(515)
graf(particion1)
plt.show() #Todavía esta medio feo. Vamos por una 3er alternativa


#Alternativa 3: elegir solamente el 5% de los datos (elegidos al azar), para poder graficarlos.
np.random.seed(999)
a=np.arange(len(sensores))
b=np.random.choice(a,int(len(sensores)/20),replace=False)
b=np.sort(b)
sensores2=sensores.iloc[b,]
print(sensores2.shape)
graf(sensores2)
plt.show() #Bastante mejor. Que pasa si probamos ahora solamente con el 1% de los datos
c=np.random.choice(a,int(len(sensores)/100),replace=False)
c=np.sort(c)
sensores3=sensores.iloc[c,]
print(sensores3.shape)
plt.figure(figsize=(25,10))
graf(sensores3) #Nos vamos a quedar con este gráfico. Entonces le agregamos algunas cosas más.
plt.xlabel("Tiempo",size=15)
plt.ylabel("Temperatura",size=15)
plt.title("Serie de tiempo con sensado de temperaturas",size=25)
plt.legend(loc="best")
plt.show()
print("\n Cantidad de datos que registran una temperatura <=15°C")
print(sensores[sensores["temp"]<=15]["temp"])


"""
De todo el análisis hecho para el feature de temperatura en este dataset, podemos obtener 3 conclusiones/aprendizajes
que van a servirnos de ahora en adelante a la hora de realizar el análisis de todos los demás features, además del
feature de temperatura que deberemos retrabajar a partir de estos aprendizajes:
1) La ausencia d enulos en el dataset hará más sencillos los análisis
2) La visualización del 100% de los datos dificulta muchísimo el análisis de los mismos. En algunos casos
seguramente sea beneficioso graficar una pequeña porción de los mismos.
3) Hay que prestar atención con los outliers. Por lo menos en el caso de la variable temperatura obtuvimos sobre
un total de 405184 tuplas, unas 288 (menos del 0.07%) con valores registrados menores o iguales a 15°C y según se observa
gráficamente estos valores con temperaturas menores a 15°C para ninguno de los 3 sensores guardan una continuidad,
sino que más bien son picos que habría que eliminar de la muestra ya que responderían más bien a fallas puntuales
en la medición. Está última conclusión deberemos tenerla en cuenta a la hora de analizar el resto de variables medidas
en este dataset, a partir de la eliminación de outliers que no tienen razón de ser.
"""


#Volvemos a analizar la tempertura sensada en los 3 puntos geográficos distintos, luego de nuestros primeros aprendizajes
#Realicemos un boxplot para entender mejor la distribución de nuestros datos
sns.boxplot(x=sensores.device,y=sensores.temp)
plt.title("Distribución de los datos por aparato")
plt.show() #Se observa que 2 de 3 sensores tienen un puñado de outliers a izquierda (valores mínimos)
dictio={}
dictio2={}
for i in sensores.device.unique():
    dictio[i]=sensores[sensores.device==i]["temp"].mean()
    dictio2[i]=sensores[sensores.device==i]["temp"].std()
sensores["limite_inf"]=sensores["device"].map(dictio)-3*sensores["device"].map(dictio2)
sensores4=sensores[sensores.temp>sensores.limite_inf]
print("Forma del nuevo dataset sin outliers:\n")
print(sensores4.shape)
graf(sensores4)
plt.xlabel("Tiempo",size=15)
plt.ylabel("Temperatura",size=15)
plt.title("Serie de tiempo con sensado de temperaturas",size=25)
plt.legend(loc="best")
plt.show() #La serie obtenida es mucho más sensata, aunque aún necesitamos sacar algo de ruido.


#Probaremos eliminar lo que queda de ruido a través de diferenciales
sensores4["Diferencial"]=sensores4.groupby("device")["temp"].diff()
sns.boxplot(x=sensores4["device"],y=sensores4["Diferencial"])
plt.title("Distribución de los diferenciales de datos por aparato")
plt.show()
sensores5=sensores4[sensores4.Diferencial>-0.5]
print("\n Forma del nuevo dataset:\n")
print(sensores5.shape)
graf(sensores5)
plt.xlabel("Tiempo",size=15)
plt.ylabel("Temperatura",size=15)
plt.title("Serie de tiempo con sensado de temperaturas",size=25)
plt.legend(loc="best")
plt.show() #Mucho mejor!!!


"""
Poder graficar una serie temporal lógica de la temperatura sensada por cada uno de los 3 aparatos, no fue algo
sencillo y constó de 4 pasos:
1) Etapa exploratoria de los datos
2) Primer refinamiento de outliers de valores en la distribución
3) Etapa exploratoria de la derivada de la serie temporal
4) Segundo refinamiento de outlier de las derivadas de la serie
Ahora se quiere hacer lo mismo para el resto de variables del dataset (humedad, monóxido de carbono, humo, etc)
pero sin necesidad de escribir 100 líneas de código para cada factor. Entonces surge la necesidad de armar 4
funciones bien robustas (una por cada uno de los pasos descriptos más arriba), con el objetivo de iterar rápidamente
a través de todas las variables.
"""


#Función para etapa exploratoria de las variables
def exploratorio(dataset,variable):
    print("\n Analisis de la variable {}\n".format(variable))
    print(dataset.groupby("device")[variable].describe())
    sns.boxplot(x=dataset["device"],y=dataset[variable])
    plt.title("Distribución de los datos de {} por aparato".format(variable))
    plt.show()
    for i in dataset["device"].unique():
        plt.plot(dataset[dataset["device"]==i]["ts"],dataset[dataset["device"]==i][variable],label=i)
    plt.xlabel("Tiempo",size=15)
    plt.ylabel("{}".format(variable),size=15)
    plt.title("Serie de tiempo con sensado de {}".format(variable),size=25)
    plt.legend(loc="best")
    plt.show()
exploratorio(sensores,"co")
for i in ["co","humidity","smoke","lpg"]:
    exploratorio(sensores,i) 
    
"""
Observaciones:
1) La serie temporal de monóxido de carbono está ok
2) La serie temporal de humedad no esta ok. Hay que avanzar por lo menos 1 paso más
3) La serie temporal lpg está ok
4) La serie temporal smoke está ok
5) Las series que hablan de calidad del aire - co, lpg y smoke - además de estar ok, guardan
todas un patrón muy similar
"""

#Función para primer refinamiento de outliers de la distribución
def refinamiento1(dataset,variable):
    dic={}
    dic2={}
    for i in dataset.device.unique():
        dic[i]=dataset[dataset.device==i][variable].mean()
        dic2[i]=dataset[dataset.device==i][variable].std()
    dataset["limite_inf"]=dataset["device"].map(dic)-3*dataset["device"].map(dic2)
    dataset2=dataset[dataset[variable]>dataset["limite_inf"]]
    print("Forma del nuevo dataset sin outliers:\n")
    print(dataset2.shape)
    for l in dataset2["device"].unique():
        plt.plot(dataset2[dataset2["device"]==l]["ts"],dataset2[dataset2["device"]==l][variable],label=l)
    plt.xlabel("Tiempo",size=15)
    plt.ylabel("{}".format(variable),size=15)
    plt.title("Serie de tiempo con sensado de {}".format(variable),size=25)
    plt.legend(loc="best")
    plt.show()
    return (dataset2)
for h in ["humidity","temp"]:
    refinamiento1(sensores,h) #Tanto para la variable humidity como para la variable temp, necesitamos refinar más


#Función para etapa exploratoria de las derivadas
def explora2(dataset,variable):
    set1=refinamiento1(dataset,variable)
    set1["Diferencial"]=set1.groupby("device")[variable].diff()
    sns.boxplot(x=set1["device"],y=set1["Diferencial"])
    plt.title("Distribución de los diferenciales de datos por aparato")
    plt.show()
    return (set1)
for j in ["humidity","temp"]:
    explora2(sensores,j)

#Función para segundo refinamiento de outliers de la derivada de la serie temporal
def refina2(dataset,variable,minimo=-1000000,maximo=1000000):
    set2=explora2(dataset,variable)
    set3=set2[(set2.Diferencial>minimo) & (set2.Diferencial<maximo)]
    for i in set3.device.unique():
        plt.plot(set3[set3["device"]==i]["ts"],set3[set3["device"]==i][variable],label=i)
    plt.xlabel("Tiempo",size=15)
    plt.ylabel("{}".format(variable),size=15)
    plt.title("Serie de tiempo FINAL con sensado de {}".format(variable),size=25)
    plt.legend(loc="best")
    plt.show()
refina2(sensores,"humidity",-5,5)
refina2(sensores,"temp",-0.5,0.5)











