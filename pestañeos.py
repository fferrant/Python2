"""
Contador de parpadeos alternativo
"""

#Importamos las librerías a utilizar
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from scipy import stats
import seaborn as sns

#Cargamos el dataset
results = []
with open('data/blinking.dat') as inputfile:
    for row in csv.reader(inputfile):
        rows = row[0].split(' ')
        results.append(rows[1:])
results=np.asarray(results) #Convertimos los valores obtenidos a un arreglo bidimensional
results = results.astype(int) #Pasamos los valores a enteros
eeg=results[:,1] #Obtenemos un vector con las mediciones tomadas del eeg 


#Obtenemos algunos datos generales de nuestra matriz result y de nuestro vector eeg
print("Primeros valores del detaset:\n")
print(results[0:10,])
print("Dimensiones de la matriz results: {}".format(results.shape))
print("Algunos valores del vector egg\n")
print("Longitud: {}".format(len(eeg)))
print("Máximo valor: {}".format(eeg.max()))
print("Mínimo valor: {}".format(eeg.min()))
print("Rango: {}".format(eeg.max()-eeg.min()))
print("Valor promedio: {}".format(eeg.mean()))
print("Varianza: {}".format(eeg.var()))
print("Desvío standard: {}".format(math.sqrt(eeg.var())))
plt.figure(figsize=(12,5))
plt.plot(eeg,color="green")
plt.ylabel("Medición",size=10)
plt.xlabel("Número de medición",size=10)
plt.title("Serie temporal de eeg",size=20)
plt.show()

#Realizamos algunas pruebas de normalidad acerca de la distribución del vector eeg

print('normality = {}'.format(scipy.stats.normaltest(eeg)))
sns.distplot(eeg)
plt.title("Supuestos de normalidad del vector eeg")
plt.show()
sns.boxplot(eeg,color="red")
plt.title("Supuestos de normalidad del vector eeg V2")
plt.show()
res = stats.probplot(eeg, plot = plt)
plt.title("Supuestos de normalidad V3") 
plt.show()
"""
De los 3 gráficos generados para probar los supuestos de normalidad, obtenemos que si bien la distribución
de los datos es simétrica, no podemos verificar normalidad ya que la distirubucíon es una de colas ligeras con mucha
nformación sobre los extremos de la distriución. Encontraremos entonces nuestros parpadeos como resultado de todos esos
outliers que están alejados de la media poblacional de nuestros datos. Definiremos entonces un umbral de +/- 3 desvíos
standard respecto a la media.
"""

#Obtenemos nuestros umbrales para distinguir un parpadeo respecto a lo que no lo es
umbral_superior=int(eeg.mean()+3*eeg.std())
print("Umbral superior: {}".format(umbral_superior))
umbral_inferior=int(eeg.mean()-3*eeg.std())
print("Umbral inferior: {}".format(umbral_inferior))
plt.figure(figsize=(12,5))
plt.plot(eeg,color="green")
plt.plot(np.full(len(eeg),umbral_superior),'r--')
plt.plot(np.full(len(eeg),umbral_inferior),'r--')
plt.ylabel("Medición",size=10)
plt.xlabel("Número de medición",size=10)
plt.title("Serie temporal de eeg con límites de control",size=20)
plt.annotate("Umbral superior",xy=(500,umbral_superior+10),color="red")
plt.annotate("Umbral inferior",xy=(500,umbral_inferior+10),color="red")
plt.show()

"""
Aplicaremos filtros a nuestros datos para transformarlos en una terna según si están por encima del umbral
superior (asignar valor 1), por debajo del umbral inferior (asignar valor -1) o entre los 2 umbrales (asignar
valor 0). Luego para determinar la cantidad de parpadeos, se contará la cantidad de ocasiones en las cuales la
serie pasa de valor cero a valor uno, es decir la cantidad de ocasiones que desde un estado de reposo las
mediciones de eeg superan el umbral superior.
"""
filtro_eeg=[]
contador=0
for i in range(len(eeg)):
    if i==0:
        filtro_eeg.append(0)
    elif eeg[i]>umbral_superior:
        filtro_eeg.append(1)
        if eeg[i-1]<=umbral_superior:
            print(i)
            contador=contador+1
    elif eeg[i]<umbral_inferior:
        filtro_eeg.append(-1)
    else:
        filtro_eeg.append(0)
print("Cantidad de parpadeos: {}".format(contador))
filtro_eeg=np.asarray(filtro_eeg)
plt.figure(figsize=(16,5))
plt.plot(filtro_eeg,color="blue")
plt.title("Filtro temporal de parpadeos",size=20)
plt.ylabel("Clase ternaria",size=10)
plt.xlabel("Número de medición",size=10)
plt.show()
