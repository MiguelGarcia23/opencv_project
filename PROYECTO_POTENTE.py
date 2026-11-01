import cv2
import os

# =================================================================================
# CONFIGURACIÓN INICIAL
# =================================================================================
archivo_foto_equipo = './imagenes/caras/dorsales.webp'  # Foto del equipo
carpeta_fotos = './imagenes/caras_prueba/'        # Carpeta donde se guardarán las caras de los jugadores
archivo_video = './videos/video_lamine.mp4'     # Video a procesar

# Crear la carpeta si no existe
if not os.path.exists(carpeta_fotos):
    os.makedirs(carpeta_fotos)

clasificador = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

color_marco = (0, 255, 0) # Marco verde
grosor = 2
fuente = cv2.FONT_HERSHEY_SIMPLEX

nombres_plantilla = []
caras_plantilla = []

# =================================================================================
# FASE 1: Crear la base de datos de caras de manera interactiva
# =================================================================================
print("--- FASE 1: REGISTRO DE JUGADORES ---")
img_equipo = cv2.imread(archivo_foto_equipo)

if img_equipo is not None:
    img_equipo_byn = cv2.cvtColor(img_equipo, cv2.COLOR_BGR2GRAY)
    caras_en_foto = clasificador.detectMultiScale(img_equipo_byn, scaleFactor=1.1, minNeighbors=5)
    
    print(f"Se han detectado {len(caras_en_foto)} caras en la foto de equipo.")
    
    for (x, y, ancho, alto) in caras_en_foto:
        
        # 1. Recortamos la cara para mostrarla y guardarla
        cara_recortada_color = img_equipo[y:y+alto, x:x+ancho]
        cara_recortada_byn = img_equipo_byn[y:y+alto, x:x+ancho]
        
        # 2. Mostramos la cara ampliada en una ventana para que el usuario pueda verla y reconocer al jugador
        cara_ampliada = cv2.resize(cara_recortada_color, (200, 200))
        cv2.imshow('Registro - ¿Quién es?', cara_ampliada)
        cv2.waitKey(1)
        
        # 3. Pedimos el nombre por consola
        nombre = input("Escribe el nombre del jugador (o presiona ENTER para ignorar): ")
        
        # 4. Si se ha escrito algo, se estandariza y se guarda en disco y memoria
        if nombre.strip() != "":
            cara_estandar = cv2.resize(cara_recortada_byn, (100, 100))
            ruta_guardado = carpeta_fotos + nombre + ".jpg"
            
            cv2.imwrite(ruta_guardado, cara_estandar)
            
            # La metemos directamente en las listas para usarlas ahora en el video
            nombres_plantilla.append(nombre)
            caras_plantilla.append(cara_estandar)
            print(f"--> ¡{nombre} guardado correctamente!")
            
    cv2.destroyWindow('Registro - ¿Quién es?')
else:
    print("No se encontró la foto del equipo. Asegúrate de que la ruta sea correcta.")

print(f"\n--- FASE 2: RECONOCIMIENTO EN VÍDEO ---")
print(f"Plantilla lista con {len(nombres_plantilla)} jugadores.")

# =================================================================================
# FASE 2: PROCESAR EL VÍDEO (basado en reproduciendoVideo.py y escalado.py)
# =================================================================================
if len(nombres_plantilla) > 0:
    video = cv2.VideoCapture(archivo_video)

    if not video.isOpened():
        print("Error al abrir el video")
        exit()

    while True:
        ret, frame = video.read()
        if not ret:
            break # Fin del video
            
        frame_byn = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Para evitar el lag en el video, se reduce la imagen para el detector
        alto, ancho = frame_byn.shape
        escala = 0.5
        ancho_esc, alto_esc = int(ancho * escala), int(alto * escala)
        frame_pequeno = cv2.resize(frame_byn, (ancho_esc, alto_esc))
        
        caras_video = clasificador.detectMultiScale(frame_pequeno, scaleFactor=1.2, minNeighbors=5)
        
        for (x_p, y_p, ancho_p, alto_p) in caras_video:
            
            # Devolver coordenadas al tamaño original
            x, y = int(x_p / escala), int(y_p / escala)
            a_cara, h_cara = int(ancho_p / escala), int(alto_p / escala)
            
            # Recorte y estandarización
            cara_vid_recortada = frame_byn[y:y+h_cara, x:x+a_cara]
            cara_vid_estandar = cv2.resize(cara_vid_recortada, (100, 100))
            
            mejor_diferencia = 1.0 
            mejor_nombre = nombres_plantilla[0]
            
            for indice in range(len(caras_plantilla)):
                patron = caras_plantilla[indice]
                
                # Comparación matemática
                res = cv2.matchTemplate(cara_vid_estandar, patron, cv2.TM_SQDIFF_NORMED)
                min_val, _, _, _ = cv2.minMaxLoc(res)
                
                if min_val < mejor_diferencia:
                    mejor_diferencia = min_val
                    mejor_nombre = nombres_plantilla[indice]
                    
            cv2.rectangle(frame, (x, y), (x+a_cara, y+h_cara), color_marco, grosor)
            cv2.putText(frame, mejor_nombre, (x, y-10), fuente, 1, color_marco, grosor)
            
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
else:
    print("No hay jugadores registrados. Fin del programa.")