import cv2
import os

DirData = 'C:/Users/JULIAN/Documents/PYTHON/Detectorface/imagenes' #ubicacion de la carpeta donde se estan las imagenes
imgPaths = os.listdir(DirData)
print('imgPath', imgPaths)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPHFace.xml') # nombre del modelo guardado xml

#configuramos la camara
camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#configuaramps el modelo haarscascade
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    x1, x2 = camara.read()
    if x1 == False: break
    gris = cv2.cvtColor(x2, cv2.COLOR_BGR2GRAY)
    auxx2 = gris.copy()

    rostro = faceClassif.detectMultiScale(gris, 1.3, 5)

    for (x,y,w,h) in rostro:
        rostros = auxx2[y:y+h, x:x+w]
        rostros = cv2.resize(rostros,(150,150), interpolation = cv2.INTER_CUBIC) #redimencionamos
        prediccion = face_recognizer.predict(rostros)

        cv2.putText(x2, '{}'.format(prediccion), (x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA) # color de la letras de prediccion

        if prediccion[1] < 90: #para determinar si es el rostro que esta en la base
            cv2.putText(x2, '{}'.format(imgPaths[prediccion[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(x2, (x,y), (x+w, y+h), (0,255,0),2)
        else:
            cv2.putText(x2,'Rostro Desconocido', (x, y-20), 2, 0.8, (0,0,255),1, cv2.LINE_AA)
            cv2.rectangle(x2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('x2', x2)
    k = cv2.waitKey(1) #para guardar
    if k == 27:
        break

camara.release()
cv2.destroyAllWindows()
