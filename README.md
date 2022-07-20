# reconocimiento-facial
reconocimiento facial mediante el modelo haarcascades

En esta proyecto de tenemos 3 scrips de python detallamoa acontinuacion su funcionamiento:

+ AgregarFace: Permite agregar un nuevo rostro realizando la captura del mismo en tiempo real
    mediante la webcam o la reproduccion de un video, se coloca el nombre a quien se va a
    identificar en la carpeta en la cual se guardaran 300 fotos para luego realizar el entrenamiento.
    
+ Entrenamiento: lee todas las carpetas que se encuentran en la carpeta imagenes la cual contiene
     todas las carpetas con las fotos de las personas a detectar, relizando un entrenamiento con el
     modelo haarcascades que se especializa en la deteccion de rostros, y crea un archivo .xml el cual 
     se utilizara en el 3er scrip para la deteccion del rostro.
     
+ detectorface: una ves que ya obtengamos el archivo xml procedemos a ejecutar el codigo, se abrira la
     webcam y detectara los rostros registrados.
     
     

