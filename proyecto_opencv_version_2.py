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
jugadores = {}
for dorsal, info in datos_equipo.items():
    info_jugador = info.copy()
    info_jugador["dorsal"] = dorsal
    jugadores[info["nombre"].lower()] = info_jugador

# =================================================================================
# CONFIGURACIÓN INICIAL
# =================================================================================
foto_equipo = './imagenes/caras/dorsales.webp'
carpeta_caras = './imagenes/caras_prueba/'
video_partido = './videos/barcelona.mp4'

# Crear la carpeta si no existe
if not os.path.exists(carpeta_caras):
    os.makedirs(carpeta_caras)

color_marco = (0, 255, 0)   # Marco verde
grosor = 2
fuente = cv2.FONT_HERSHEY_SIMPLEX

# --- AJUSTES DE VELOCIDAD ---
ANCHO_IA = 480          # Ancho (en píxeles) de la imagen que analiza la IA. Menos = más rápido
CADA_N_FRAMES = 3       # La IA analiza 1 de cada N fotogramas. Más = más rápido
CADA_N_BARRA = 10       # La barra de tiempo se actualiza 1 de cada N fotogramas
NOMBRE_VENTANA = 'Reconocimiento de jugadores - Version final'

# =================================================================================
# FASE 1: IDENTIFICACIÓN DE JUGADORES
# =================================================================================
print("--- FASE 1: REGISTRO DE JUGADORES ---")
img_equipo = cv2.imread(foto_equipo)

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
            for dorsal_disponible, nombre_disponible in jugadores_disponibles.items():
                print(f"{dorsal_disponible}. {nombre_disponible}")

            jugador_seleccionado = input("\nEscribe el NÚMERO del jugador (o presiona ENTER para ignorar): ").strip()

            if jugador_seleccionado == "":
                print("Cara ignorada.")
                break  # Sale del bucle de validación y pasa a la siguiente cara

            elif jugador_seleccionado in jugadores_disponibles:
                # 3. Si acierta el número, sacamos el nombre exacto del JSON
                nombre_exacto = jugadores_disponibles[jugador_seleccionado]

                ruta_guardado = os.path.join(carpeta_caras, nombre_exacto + ".jpg")
                cv2.imwrite(ruta_guardado, cara_recortada)
                print(f"--> ¡{nombre_exacto} registrado con éxito!")

                # 4. Lo borramos de la lista de disponibles para que no vuelva a salir
                del jugadores_disponibles[jugador_seleccionado]
                break  # Sale del bucle de validación y pasa a la siguiente cara

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
for archivo in os.listdir(carpeta_caras):
    if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
        ruta = os.path.join(carpeta_caras, archivo)
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
    video = cv2.VideoCapture(video_partido)

    if not video.isOpened():
        print("Error al abrir el vídeo")
        exit()

    # --- CONFIGURACIÓN DEL REPRODUCTOR ---
    # 1. Averiguamos cuántos fotogramas tiene el vídeo en total
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # 2. Es obligatorio crear la ventana antes de pegarle la barra
    cv2.namedWindow(NOMBRE_VENTANA)

    # Interruptor para saber si la barra la mueve el programa o el usuario
    barra_movida_por_programa = False

    # Variables usadas para mejorar el lag
    contador_frames = 0
    posiciones_caras = []
    nombres_caras = []

    # 3. Función que se dispara cuando se mueve la barra
    def saltar_a_fotograma(valor):
        global contador_frames, posiciones_caras, nombres_caras
        # Si la ha movido el propio programa, NO saltamos
        if barra_movida_por_programa:
            return
        # Si la ha movido el usuario con el ratón, saltamos a ese punto del vídeo
        video.set(cv2.CAP_PROP_POS_FRAMES, valor)
        # Borramos los recuadros antiguos y forzamos que la IA analice el siguiente frame
        posiciones_caras, nombres_caras = [], []
        contador_frames = 0

    # 4. Creamos la barra de desplazamiento pegada a nuestra ventana
    cv2.createTrackbar('Tiempo', NOMBRE_VENTANA, 0, max(total_frames - 1, 1), saltar_a_fotograma)
    # --------------------------------------------

    escala = None  # Se calcula con el primer fotograma

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Escala automática para que la IA siempre trabaje con una imagen de ANCHO_IA píxeles
        if escala is None:
            escala = min(1.0, ANCHO_IA / frame.shape[1])

        # La barra va avanzando (solo cada CADA_N_BARRA fotogramas)
        if contador_frames % CADA_N_BARRA == 0:
            frame_actual = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            barra_movida_por_programa = True
            cv2.setTrackbarPos('Tiempo', NOMBRE_VENTANA, frame_actual)
            barra_movida_por_programa = False

        # Solo hacemos el cálculo pesado 1 de cada CADA_N_FRAMES fotogramas
        if contador_frames % CADA_N_FRAMES == 0:
            # Reducimos la imagen para que la IA vaya más rápida
            frame_pequeno = cv2.resize(frame, (0, 0), fx=escala, fy=escala)
            frame_rgb_pequeno = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2RGB)

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

        contador_frames += 1

        # =================================================================================
        # TARJETA DE ESTADÍSTICAS
        # =================================================================================
        for (top, right, bottom, left), nombre in zip(posiciones_caras, nombres_caras):

            # Devolvemos las coordenadas al tamaño original
            top = int(top / escala)
            right = int(right / escala)
            bottom = int(bottom / escala)
            left = int(left / escala)

            color = (0, 255, 0) if nombre != "Desconocido" else (0, 0, 255)

            # Dibujamos el marco de la cara siempre
            cv2.rectangle(frame, (left, top), (right, bottom), color, grosor)

            if nombre != "Desconocido":
                # 1. Buscamos al jugador en nuestra base de datos (ignorando mayúsculas/minúsculas)
                datos = jugadores.get(nombre.lower())

                if datos:
                    # 2. Dibujamos el fondo oscuro de la tarjeta a la derecha de la cara
                    ancho_tarjeta = 230
                    alto_tarjeta = 110 if "nota" not in datos else 130

                    # Fondo gris oscuro
                    cv2.rectangle(frame, (right, top), (right + ancho_tarjeta, top + alto_tarjeta), (40, 40, 40), cv2.FILLED)
                    # Borde verde para la tarjeta
                    cv2.rectangle(frame, (right, top), (right + ancho_tarjeta, top + alto_tarjeta), color, 1)

                    # 3. Preparamos los textos que van dentro
                    lineas_texto = [
                        f"#{datos['dorsal']} {datos['nombre']}",
                        f"Posición: {datos['posicion']}",
                        f"Nacionalidad: {datos['nacionalidad']}",
                        f"Edad: {datos['edad']} años"
                    ]
                    if "nota" in datos:
                        lineas_texto.append(f"*** {datos['nota']} ***")

                    # 4. Escribimos cada línea
                    y_texto = top + 25
                    for linea in lineas_texto:
                        cv2.putText(frame, linea, (right + 10, y_texto), fuente, 0.5, (255, 255, 255), 1)
                        y_texto += 20
                else:
                    # Si el nombre detectado no está en el JSON, solo ponemos el nombre
                    cv2.putText(frame, nombre, (left, top - 10), fuente, 0.75, color, grosor)
            else:
                # Si es un desconocido, recuadro rojo clásico
                cv2.putText(frame, "Desconocido", (left, top - 10), fuente, 0.75, color, grosor)

        cv2.imshow(NOMBRE_VENTANA, frame)

        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
else:
    print("No hay jugadores registrados. Fin del programa.")