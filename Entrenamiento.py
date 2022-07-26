import cv2
import os
import numpy as np

DirData = 'C:/Users/JULIAN/Documents/PYTHON/Detector/imagenes'
ListaFace = os.listdir(DirData)
print('Lista de Face: ', ListaFace)
#conteos
labels = []
rostrosData = []
label = 0

for nameDir in ListaFace:
	FacePath = DirData + '/' + nameDir
	print('Leyendo las imágenes')

	for fileName in os.listdir(FacePath):
		print('Rostros: ', nameDir + '/' + fileName)
		labels.append(label) #variable contadora

		rostrosData.append(cv2.imread(FacePath + '/' + fileName, 0))
		img = cv2.imread(FacePath+'/'+fileName,0)
		#cv2.imshow('img',img) #para ver si funciona
		#cv2.waitKey(10)
	label = label + 1 # variable de conteo

#cv2.destroyAllWindows() ##fin para ver si funciona + prints

#print('labels= ',labels) # para asegurar
#print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
#print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))

face_recognizer = cv2.face.LBPHFaceRecognizer_create() #modelos para entrenar
print("Se esta realizando el entrenamiento....")
face_recognizer.train(rostrosData, np.array(labels))
face_recognizer.write('modeloLBPHFace.xml') # nuestro modelo
print("El modelo se a guardado")
