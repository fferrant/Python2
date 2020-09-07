
"""
En este programa se creará una función que elija una imagen dentro de un menú de opciones y reciba también la
instrucción de apagar 1 o 2 de los canales de colores RGB. Entonces tendremos 6 opciones para mostrar esta imagen
"decolorada": roja, verde, azul, roja+verde, roja+azul o verde+azul. Finalmente se mostrará esta imagen "decolorada"
tanto en colores como en escala de grises.
"""


#Comenzamos cargando las librerias que necesitará nuestro programa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import image
import PIL


#Generamos los diccionario a partir de los cuales elegiremos una de las imágenes disponibles y una de las paletas de colores
diccionario={1:"images/avengers.jpg",
             2:"images/guason.jpg",
             3:"images/marvel.jpg",
             4:"images/sonic.jpg"}
diccionario2={1:[1,0,0],
              2:[0,1,0],
              3:[0,0,1],
              4:[1,1,0],
              5:[1,0,1],
              6:[0,1,1]}


#A continuación agregaremos 3 funciones que nos ayudarán a hacer el programa más compacto


#Función para obtener una matriz de grises
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

#Función para imprimir las imágenes en colores obtenidas 
def Imprimir(foto,texto):
    plt.figure(figsize=(15,15))
    plt.imshow(foto)
    plt.axis("off")
    plt.title(texto,size=25)
    plt.show()

#Función para imprimir las imágenes en escala de grises 
def Imprimir_grises(foto,texto):
    plt.figure(figsize=(15,15))
    plt.imshow(foto,cmap=plt.cm.gray)
    plt.axis("off")
    plt.title(texto,size=25)
    plt.show()


#Ahora si podemos meternos en la función que decolora las imágenes
def decolorear(opcion,paleta):
    Imagen=Image.open(diccionario[opcion]) #Cargamos la imagen origina
    Imprimir(Imagen,"Imagen original") #La imprimimos
    matriz_colores=np.asarray(Imagen,dtype=np.float32)/255 #La pasamos a una matriz de numpy
    matriz_gris=rgb2gray(matriz_colores) #Armamos nuestra matriz de grises
    Imprimir_grises(matriz_gris,"Original en escala de grises") #La imprimimos
    #A partir de este punto empezamos a trabajar en la transformación de la image a 1 o 2 canales de colores
    matriz_roja=matriz_colores[:,:,0]*diccionario2[paleta][0] #Canal rojo
    matriz_verde=matriz_colores[:,:,1]*diccionario2[paleta][1] #Canal verde
    matriz_azul=matriz_colores[:,:,2]*diccionario2[paleta][2] #Canal azul
    matriz_roja=matriz_roja[:,:,np.newaxis]
    matriz_verde=matriz_verde[:,:,np.newaxis]
    matriz_azul=matriz_azul[:,:,np.newaxis]
    FINAL=np.concatenate((matriz_roja,matriz_verde,matriz_azul),axis=2)
    Imprimir(FINAL,"Imagen decolorada")
    FINAL_GRISES=rgb2gray(FINAL)
    Imprimir_grises(FINAL_GRISES,"Imagen decolorada en grises")
    #Por último guardamos nuestra nueva imagen decolorada y en escala de grises
    guardar_decolorada = np.floor(FINAL*255)
    Image.fromarray(guardar_decolorada.astype(np.uint8)).save("images/Decolorada.jpg")
    guardar_gris = np.floor(FINAL_GRISES*255)
    Image.fromarray(guardar_gris.astype(np.uint8)).save("images/Escala_grises.jpg")


#Hagamos algunas pruebas de funcionamiento
decolorear(3,6)
decolorear(4,2)


# Finalmente completamos el programa para que sea interactivo con el usuario
uno=input("Seleccione la imagen que desea comprimir:\n\
          Opción 1: Avengers\n\
          Opción 2: Guasón\n\
          Opción 3: Marvel\n\
          Opción 4: Sonic\n")
uno=int(uno)
dos=input("Ingrese por favor la paleta de colores a utilizar:\n\
          Opción 1: Rojo\n\
          Opción 2: Verde\n\
          Opción 3: Azul\n\
          Opción 4: Rojo+Verde\n\
          Opción 5: Rojo+Azul\n\
          Opción 6: Verde+Azul\n")
dos=int(dos)
decolorear(uno,dos)


