import cv2
import face_recognition
import numpy as np
import os
import json

# =================================================================================
# CARGA DE JSON
# =================================================================================
archivo_json = './utils/barcelona.json'

try:
    with open(archivo_json, 'r', encoding='utf-8') as f:
        datos_equipo = json.load(f)
except FileNotFoundError:
    print(f"Error: No se ha encontrado el archivo {archivo_json}")
    datos_equipo = {}

# Transformamos el diccionario para poder buscar directamente por el nombre
buscador_jugadores = {}
for dorsal, info in datos_equipo.items():
    info_completa = info.copy()
    info_completa["dorsal"] = dorsal
    buscador_jugadores[info["nombre"].lower()] = info_completa

# =================================================================================
# CONFIGURACIÓN INICIAL
# =================================================================================
archivo_foto_equipo = './imagenes/caras/squad_b.jpg'
carpeta_fotos = './imagenes/caras_prueba/'
archivo_video = './videos/video_lamine.mp4'

if not os.path.exists(carpeta_fotos):
    os.makedirs(carpeta_fotos)

color_marco = (0, 255, 0)
grosor = 2
fuente = cv2.FONT_HERSHEY_SIMPLEX

# =================================================================================
# FASE 1: IDENTIFICACIÓN DE JUGADORES
# =================================================================================
print("--- FASE 1: REGISTRO DE JUGADORES ---")
img_equipo = cv2.imread(archivo_foto_equipo)

if img_equipo is not None:
    # Usa RGB, en lugar de BGR
    img_equipo_rgb = cv2.cvtColor(img_equipo, cv2.COLOR_BGR2RGB)

    # Detecta caras usando el modelo HOG de la IA (mucho más preciso que HaarCascade)
    caras_en_foto = face_recognition.face_locations(img_equipo_rgb)
    
    print(f"Se han detectado {len(caras_en_foto)} caras.")
    
    # 1. Creamos un diccionario temporal con los jugadores que faltan por asignar
    jugadores_disponibles = {dorsal: info["nombre"] for dorsal, info in datos_equipo.items()}
    
    for (top, right, bottom, left) in caras_en_foto:
        cara_recortada = img_equipo[top:bottom, left:right]
        cara_ampliada = cv2.resize(cara_recortada, (200, 200))
        
        cv2.imshow('Registro - Quien es?', cara_ampliada)
        cv2.waitKey(1)
        
        # 2. Bucle de validación: se repite hasta que el usuario meta una opción correcta
        while True:
            print("\n--- JUGADORES SIN IDENTIFICAR ---")
            # Imprimimos la lista actualizada
            for dorsal_disp, nombre_disp in jugadores_disponibles.items():
                print(f"{dorsal_disp}. {nombre_disp}")
                
            seleccion = input("\nEscribe el NÚMERO del jugador (o presiona ENTER para ignorar): ").strip()
            
            if seleccion == "":
                print("Cara ignorada.")
                break # Sale del bucle de validación y pasa a la siguiente cara
                
            elif seleccion in jugadores_disponibles:
                # 3. Si acierta el número, sacamos el nombre exacto del JSON
                nombre_exacto = jugadores_disponibles[seleccion]
                
                ruta_guardado = os.path.join(carpeta_fotos, nombre_exacto + ".jpg")
                cv2.imwrite(ruta_guardado, cara_recortada)
                print(f"--> ¡{nombre_exacto} registrado con éxito!")
                
                # 4. Lo borramos de la lista de disponibles para que no vuelva a salir
                del jugadores_disponibles[seleccion]
                break # Sale del bucle de validación y pasa a la siguiente cara
                
            else:
                # Si mete una letra o un número que no está en la lista
                print("ERROR: Número no válido o jugador ya asignado. Inténtalo de nuevo.")
            
    cv2.destroyWindow('Registro - Quien es?')
else:
    print("No se encontró la foto del equipo.")

# =================================================================================
# GENERANDO EL ADN DE CADA JUGADOR PARA APLICAR LA IA
# =================================================================================
print("\nCodificando el ADN facial de los jugadores...")
nombres_conocidos = []
codificaciones_conocidas = []

# La IA lee las fotos guardadas y extrae los 128 puntos geométricos (encodings)
for archivo in os.listdir(carpeta_fotos):
    if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
        ruta = os.path.join(carpeta_fotos, archivo)
        nombre = archivo.rsplit(".", 1)[0]
        
        # Cargar imagen y extraer sus 128 mediciones
        img_jugador = face_recognition.load_image_file(ruta)
        
        # A veces la foto no se recorta perfecta, comprobamos si encuentra la cara
        encodings = face_recognition.face_encodings(img_jugador)
        if len(encodings) > 0:
            codificaciones_conocidas.append(encodings[0])
            nombres_conocidos.append(nombre)

print(f"Plantilla lista con {len(nombres_conocidos)} jugadores.")

# =================================================================================
# FASE 2: RECONOCIMIENTO EN VÍDEO CON IA
# =================================================================================
if len(nombres_conocidos) > 0:
    video = cv2.VideoCapture(archivo_video)

    # --- Para mejorar el lag ---
    procesar_este_frame = True
    posiciones_caras = []
    nombres_caras = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Para mejorar el lag, reducir imagen a 1/4 para que la IA vaya en tiempo real
        escala = 0.25
        frame_pequeno = cv2.resize(frame, (0, 0), fx=escala, fy=escala)
        frame_rgb_pequeno = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2RGB)
        
        # Solo hacemos el cálculo pesado 1 de cada 2 fotogramas
        if procesar_este_frame:
            posiciones_caras = face_recognition.face_locations(frame_rgb_pequeno)
            codificaciones_caras = face_recognition.face_encodings(frame_rgb_pequeno, posiciones_caras)

            # Vaciamos la lista para actualizarla
            nombres_caras = []
            
            for codificacion in codificaciones_caras:
                distancias = face_recognition.face_distance(codificaciones_conocidas, codificacion)
                mejor_nombre = "Desconocido"
                
                if len(distancias) > 0:
                    indice_ganador = np.argmin(distancias)
                    # Si la distancia es menor a 0.6, es él.
                    if distancias[indice_ganador] < 0.6:
                        mejor_nombre = nombres_conocidos[indice_ganador]
                        
                nombres_caras.append(mejor_nombre)

        # Cambiamos el interruptor para que el SIGUIENTE fotograma la IA descanse
        procesar_este_frame = not procesar_este_frame
        
        # DIBUJAMOS SIEMPRE (usemos los datos recién calculados o los guardados del frame anterior)
        for (top, right, bottom, left), nombre in zip(posiciones_caras, nombres_caras):
            
            # Devolvemos las coordenadas al tamaño original
            top = int(top / escala)
            right = int(right / escala)
            bottom = int(bottom / escala)
            left = int(left / escala)
            
            color = (0, 255, 0) if nombre != "Desconocido" else (0, 0, 255)
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, grosor)
            cv2.putText(frame, nombre, (left, top - 10), fuente, 0.75, color, grosor)
            
        cv2.imshow('Reconocimiento de jugadores - Version rapida de face_recognition', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()