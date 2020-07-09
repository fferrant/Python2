
  
"""3) Arme un diccionario en donde cada clave sea el nombre de la columna y cada valor sea una lista con los valores de esa columna"""
 
"""4) Convierta los string numericos, en datos tipo float()"""

"""5) Arme una funcion que dado el nombre de la columna y el diccionario del punto anterior, calcule la media y el desvio estandar de los datos ,
 ambos valores, media y desvio deben ser devueltos en una tupla = (media, desvio)"""
 

import sys
import pprint
import psycopg2

con=psycopg2.connect(host='localhost', user='postgres', password='delaburo2012', database='Ventas_eco')
cursor=con.cursor()

cursor.execute("INSERT INTO city(cityname,citykey) VALUES ('Buenos Aires',1150);")
con.commit()
cursor.execute("select * from city where cityname='Buenos Aires';")

registros=cursor.fetchall()

pprint.pprint(registros)


cursor.close()
con.close()




    


       




    










 
