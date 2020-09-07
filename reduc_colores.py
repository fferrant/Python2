
"""
En este programa crearemos una función que en primer lugar elija una imagen de un menú de 4 opciones disponibles.
En segundo lugar indicaremos a la función un factor de reducción de colores. Por ejemplo si a una imagen con 700
colores diferentes, le aplicamos un filtro del 50%, debería devolvernos una imagen que solamente utilice 350 colores.
Para lograr esto, aplicaremos a la imagen, separándola en pixel por pixel, un algoritmo de Kmeans. Los colores con
que se mostrará la imagen, serán los centros de cada cluster.
"""


#En primer lugar importamos las librerías que neceitamos para este programa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
import random
import image
import PIL
from PIL import Image


#Creamos un diccionario para luego seleccionar la imagen a reducir
diccionario={1:"images/avengers.jpg",
             2:"images/guason.jpg",
             3:"images/marvel.jpg",
             4:"images/sonic.jpg"}


#Luego procedemos a crear la función que hace la reducción de la imagen seleccionada
def reducidor(col,opcion):
    imagen=Image.open(diccionario[opcion]) #Cargamos la imagen con la que vamos a trabajar
    print("\n\nParámetros de nuestra imagen\n") #Aquí comenzamos a obtener parámetros de la imagen
    print("Dimensiones de la imagen: {}".format(imagen.size))
    fila,columna=imagen.size
    cant_colores=len(imagen.getcolors(fila*columna))
    print("Cantidad de píxeles: {}".format(fila*columna))
    print("Cantidad de colores: {}".format(cant_colores))
    colores_nuevos=col
    print("Cantidad de colores a los cuales reduciremos la imagen:{}".format(colores_nuevos)) #Aqui finaliza la obtención de parámetros
    #A continuación trabajaremos en la preparación de los datos para la aplicación del algoritmo
    matriz=np.asarray(imagen,dtype=np.float32)/255 #Obtenemos una matriz de la imagen
    print("\n Dimensiones de la matriz original: {}".format(matriz.shape)) #LA 3ra dimensión son los canales de colores
    R=matriz[:,:,0] #Separamos la matriz en los 3 canales R,G,B
    G=matriz[:,:,1]
    B=matriz[:,:,2]
    #Ahora convertiremos cada sub-matriz en un vector columna
    RED=R.reshape((-1,1))
    GREEN=G.reshape((-1,1))
    BLUE=B.reshape((-1,1))
    #Luego concatenamos los 3 vectores columna para formar la nueva matriz sobre la que aplicaremos Kmeans
    X=np.concatenate((RED,GREEN,BLUE),axis=1)
    print("Dimensiones de la matriz reconvertida: {}".format(X.shape))
    #Ahora si aplicamos el algoritmo de Kmeans
    random.seed(0)
    modelo=KMeans(n_clusters=colores_nuevos,random_state=0).fit(X)
    grupos=modelo.labels_ #Obtenemos los grupos de cada pixel
    centroides=modelo.cluster_centers_ #El nuevo color estará representado por el centroide de cada grupo
    rojo=[]
    verde=[]
    azul=[]
    #Ahora generaremos la matriz con la nueva imagen a partir de los colores que son el centroide de cada grupo
    for i in range(len(X[:,0])): 
        rojo.append(centroides[grupos[i]][0])
        verde.append(centroides[grupos[i]][1])
        azul.append(centroides[grupos[i]][2])
    rojo=np.asarray(rojo)
    verde=np.asarray(verde)
    azul=np.asarray(azul)
    matriz_roja=rojo.reshape((columna,fila))
    matriz_verde=verde.reshape((columna,fila))
    matriz_azul=azul.reshape((columna,fila))
    matriz_roja = matriz_roja[:, :, np.newaxis]
    matriz_verde = matriz_verde[:, :, np.newaxis]
    matriz_azul = matriz_azul[:, :, np.newaxis]
    matriz_final=np.concatenate((matriz_roja,matriz_verde,matriz_azul),axis=2)
    print("Forma de la matriz reducida: {}".format(matriz_final.shape))
    #Ahora podemos proceder a imprimir las 2 imágenes: real vs reducida
    #Primero la imagen real
    plt.figure(figsize=(15,15))
    plt.imshow(matriz)
    plt.axis('off')
    plt.title("Imagen original ({} colores)".format(cant_colores))
    plt.show()
    #...Y luego la imagen reducida
    plt.figure(figsize=(15,15))
    plt.imshow(matriz_final)
    plt.axis('off')
    plt.title("Imagen reducida ({} colores)".format(colores_nuevos))
    plt.show()
    #Como último paso, guardamos la imagen en un nuevo archivo
    imagen_final = np.floor(matriz_final*255)
    Image.fromarray(imagen_final.astype(np.uint8)).save("images/colores_reducidos.jpg")


#Ahora que tenemos la función creada, solicitaremos los parámetros para utilizarla de forma interactiva
uno=input("Seleccione la imagen que desea comprimir:\n\
          Opción 1: Avengers\n\
          Opción 2: Guasón\n\
          Opción 3: Marvel\n\
          Opción 4: Sonic\n")
uno=int(uno)
dos=input("Ingrese la cantidad de colores a las cuales desea reducir la imagen:\t")
dos=int(dos)
reducidor(dos,uno)
