
"""
En este programa tomaremos la imagen original de un conejo y la convertiremos en la misma imagen pero mostrando
en un único color (que puede ser rojo, azul o verde) al conejo y en otro color distinto (que puede ser azul,
rojo o verde) el fondo de la imagen. Para realizar esto se realizará una clusterización de los píxeles de la
imagen con 2 clusters. A los píxeles de uno de los grupos se les activará uno de los 3 canales RGB y al segundo
grupo uno de los 2 canales restantes. Esto estará enmarcado en una función interactiva con el usuario que le
permita a este elegir que 2 colores utilizar.
"""


#Comenzamos cargando las librerías a utilizar
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import random
import image
from sklearn.cluster import KMeans
import time


#Ahora cargamos la imagen a utilizar y obtenemos algunos parámetros de la misma
imagen=Image.open("images/conejo.jpg") #Cargamos la imagen con la que vamos a trabajar
print("\n\nParámetros de nuestra imagen\n") #Aquí comenzamos a obtener parámetros de la imagen
print("Dimensiones de la imagen: {}".format(imagen.size))
fila,columna=imagen.size
print("Cantidad de píxeles de la imagen: {}".format(fila*columna))
cant_colores=len(imagen.getcolors(fila*columna))
print("Cantidad de colores de la imagen: {}".format(cant_colores))


#Imprimimos la imagen original
plt.figure(figsize=(15,15)) 
plt.imshow(imagen)
plt.axis("off")
plt.title("Imagen original",size=20)
plt.show()


#Ahora generamos la función que crea la imagen bicolor
def convertidor(opcion):
    matriz=np.asarray(imagen,dtype=np.float32)/255 #Obtenemos una matriz de la imagen
    print("\n Dimensiones de la matriz original: {}".format(matriz.shape)) #LA 3ra dimensión son los canales de colores
    #Separamos la matriz en los 3 canales R,G,B
    ROJO=matriz[:,:,0]
    VERDE=matriz[:,:,1]
    AZUL=matriz[:,:,2]
    #Ahora convertimos cada una de estas 3 matrices en un vector columna
    RED=ROJO.reshape((-1,1))
    GREEN=VERDE.reshape((-1,1))
    BLUE=AZUL.reshape((-1,1))
    #Obtenemos la matriz sobre la cual aplicarmeos el algoritmo de Kmeans
    X=np.concatenate((RED,GREEN,BLUE),axis=1)
    #Procedemos a realizar la clusterización
    random.seed=0
    modelo=KMeans(n_clusters=2,random_state=0).fit(X)
    #Las líneas que siguen sirven para asignar cada pixel a un canal RGB
    UNO=[]
    DOS=[]
    for i in modelo.labels_:
        if i==0:
            UNO.append(0)
            DOS.append(1)
        else:
            UNO.append(1)
            DOS.append(0)
    UNO=np.asarray(UNO)
    DOS=np.asarray(DOS)
    ONE=UNO.reshape((columna,fila))
    TWO=DOS.reshape((columna,fila))
    THREE=np.zeros((columna,fila))
    ONE=ONE[:,:,np.newaxis]
    TWO=TWO[:,:,np.newaxis]
    THREE=THREE[:,:,np.newaxis]
    #Con esto confeccionamos la matriz de la imagen reconvertida
    if opcion==1:    
        Z=np.concatenate((ONE,TWO,THREE),axis=2)
    elif opcion==2:
        Z=np.concatenate((THREE,ONE,TWO),axis=2)
    elif opcion==3:
        Z=np.concatenate((TWO,THREE,ONE),axis=2)        
    #Ahora imprimimos la imagen reconvertida en colores
    plt.figure(figsize=(15,15))
    plt.imshow(Z)
    plt.axis("off")
    plt.title("Imagen bicolor",size=20)
    plt.show()
    #Y luego guardamos la imagen
    imagen_final = np.floor(Z*255)
    Image.fromarray(imagen_final.astype(np.uint8)).save("images/bicolor.jpg")



#Realizamos ahora algunas impresiones
convertidor(1) #Imagen roja y verde
convertidor(2) #Imagen verde y azul
convertidor(3) #Imagen azul y roja


#Por último dejamos un input para que un usuario pueda elegir la combinación de colores
comb=input("Ingrese la combinación de colores deseada:\n\
            1. Imagen roja y verde\n\
            2. Imagen verde y azul\n\
            3. Imagen azul y roja\n")
try:
    comb=int(comb)
except:
    print("Tipo de variable inadecuada\n")
while comb not in [1,2,3]:
    if comb in [1,2,3]:
        pass
    else:
        comb=input("Ingrese la combinación de colores deseada:\n\
            1. Imagen roja y verde\n\
            2. Imagen verde y azul\n\
            3. Imagen azul y roja\n")
        try:
            comb=int(comb)
        except:
            print("Tipo de variable inadecuada\n")
convertidor(comb)
