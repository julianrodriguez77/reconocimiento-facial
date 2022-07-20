import cv2
import os
import imutils

#el nombre de la carpeta donde se guardan lan nuevas fotos
CarpName = 'BenAflec'
DirData = 'C:/Users/JULIAN/Documents/PYTHON/DetectorAnimal/imagenes'
AnimalPath = DirData + '/' + CarpName  #direccion y nombre le la nueva carpeta
#condicional para saber la existenia de la carpeta
if not os.path.exists(AnimalPath):
    print('SE CREO LA CARPETA:', AnimalPath)
    os.makedirs(AnimalPath)
#conectamos la camara
camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#contador para contar el numero de fotos que se almancenan
#haarscascade modelo para e
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0
#Bucle
while True:
    x1, x2 = camara.read() #Creamo variables para la camara x2 almasena imagen
    if x1 == False:
        break
    x2 = imutils.resize(x2, width=320)#redimencionar la pantalla
    gris = cv2.cvtColor(x2, cv2.COLOR_BGR2GRAY)#escala de grises las imagenes
    auxx2 = x2.copy() #copia del video

    rostros = faceClassif.detectMultiScale(gris, 1.3, 5) # clasifica en escalas de grises

    for (x, y, w, h) in rostros:
        cv2.rectangle(x2, (x,y), (x+w, y+h), (0, 255, 0), 2) #rectangulo detector
        rostros2 = auxx2[y:y+h, x:x+w]
        rostros2 = cv2.resize(rostros2, (720, 720), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(AnimalPath + '/rostro2_{}.jpg'.format(count),rostros2)
        count = count + 1


    cv2.imshow('x2', x2) #pantalla de video

    k = cv2.waitKey(1) #el 27 representa la tecla esc para cerrar la app
    if k==27 or count >= 300:
        break

camara.release()
cv2.destroyAllWindows



