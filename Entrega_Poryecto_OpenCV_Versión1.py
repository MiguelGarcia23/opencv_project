import cv2
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
archivo_foto_equipo = './imagenes/caras/dorsales.webp'  # Foto del equipo
carpeta_fotos = './imagenes/caras_prueba/'        # Carpeta donde se guardarán las caras de los jugadores
archivo_video = './videos/barcelona.mp4'

# Crear la carpeta si no existe
if not os.path.exists(carpeta_fotos):
    os.makedirs(carpeta_fotos)

clasificador = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

# Comprobamos que el detector de caras se ha cargado bien
if clasificador.empty():
    print("ERROR: no se pudo cargar el detector de caras (revisa la carpeta haarcascades).")
    exit()

color_marco = (0, 255, 0) # Marco verde
grosor = 2
fuente = cv2.FONT_HERSHEY_SIMPLEX
UMBRAL = 0.3 # Diferencia máxima para aceptar un nombre (0 = idénticas). Si se supera, "Desconocido"

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

    # 1. Creamos un diccionario temporal con los jugadores que faltan por asignar
    jugadores_disponibles = {dorsal: info["nombre"] for dorsal, info in datos_equipo.items()}
        
    for (x, y, ancho, alto) in caras_en_foto:
        
        # 1. Recortamos la cara para mostrarla y guardarla
        cara_recortada_color = img_equipo[y:y+alto, x:x+ancho]
        cara_recortada_byn = img_equipo_byn[y:y+alto, x:x+ancho]
        
        # 2. Mostramos la cara ampliada en una ventana para que el usuario pueda verla y reconocer al jugador
        cara_ampliada = cv2.resize(cara_recortada_color, (200, 200))
        cv2.imshow('Registro - ¿Quién es?', cara_ampliada)
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

                cara_estandar = cv2.resize(cara_recortada_byn, (100, 100))
                cara_estandar = cv2.equalizeHist(cara_estandar) # Igualamos el contraste para que la luz influya menos
                ruta_guardado = os.path.join(carpeta_fotos, nombre_exacto + ".jpg")
                cv2.imwrite(ruta_guardado, cara_estandar)

                # La metemos directamente en las listas para usarlas ahora en el video
                nombres_plantilla.append(nombre_exacto)
                caras_plantilla.append(cara_estandar)
                print(f"--> ¡{nombre_exacto} guardado correctamente!")

                # 4. Lo borramos de la lista de disponibles para que no vuelva a salir
                del jugadores_disponibles[seleccion]
                break # Sale del bucle de validación y pasa a la siguiente cara
            

            else:
                # Si mete una letra o un número que no está en la lista
                print("ERROR: Número no válido o jugador ya asignado. Inténtalo de nuevo.")
            
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
            cara_vid_estandar = cv2.equalizeHist(cara_vid_estandar) # Mismo tratamiento que en el registro
            
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

            # Si la diferencia es demasiado grande, no se parece a nadie
            if mejor_diferencia > UMBRAL:
                mejor_nombre = "Desconocido"

            # Mostramos la diferencia debajo de la cara (amarillo) para ajustar el UMBRAL
            cv2.putText(frame, f"{mejor_diferencia:.2f}", (x, y + h_cara + 25), fuente, 0.8, (0, 255, 255), 2)

            # =================================================================================
            # TARJETA DE ESTADÍSTICAS
            # =================================================================================
            cv2.rectangle(frame, (x, y), (x+a_cara, y+h_cara), color_marco, grosor)

            datos = buscador_jugadores.get(mejor_nombre.lower())

            if datos:
                # Coordenada derecha de la cara para pegar ahí la tarjeta
                lado_derecho = x + a_cara
                
                ancho_tarjeta = 230
                alto_tarjeta = 110 if "nota" not in datos else 130

                # Si la tarjeta no cabe a la derecha, la ponemos a la izquierda de la cara
                if lado_derecho + ancho_tarjeta > frame.shape[1]:
                    lado_derecho = max(0, x - ancho_tarjeta)
                
                # Fondo gris y borde
                cv2.rectangle(frame, (lado_derecho, y), (lado_derecho + ancho_tarjeta, y + alto_tarjeta), (40, 40, 40), cv2.FILLED)
                cv2.rectangle(frame, (lado_derecho, y), (lado_derecho + ancho_tarjeta, y + alto_tarjeta), color_marco, 1)
                
                # Textos
                lineas_texto = [
                    f"#{datos['dorsal']} {datos['nombre']}",
                    f"Posicion: {datos['posicion']}",
                    f"Nacionalidad: {datos['nacionalidad']}",
                    f"Edad: {datos['edad']} anos"
                ]
                if "nota" in datos:
                    lineas_texto.append(f"*** {datos['nota']} ***")
                    
                # Escribimos las líneas
                y_texto = y + 25
                for linea in lineas_texto:
                    cv2.putText(frame, linea, (lado_derecho + 10, y_texto), fuente, 0.5, (255, 255, 255), 1)
                    y_texto += 20
            else:
                # Fallback por si acaso no encuentra el nombre (solo dibuja el nombre arriba)
                cv2.putText(frame, mejor_nombre, (x, y-10), fuente, 1, color_marco, grosor)
            
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
else:
    print("No hay jugadores registrados. Fin del programa.")