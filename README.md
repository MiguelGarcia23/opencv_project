# ⚽ Reconocimiento de jugadores en vídeo - Versión 1 con OpenCV

Proyecto de la asignatura **Imagen Digital** (Grado en Ingeniería Informática en Ingeniería del Software, Universidad de Extremadura).

## ❓ ¿Por qué este proyecto?

Desde la grada de un estadio no siempre es fácil saber quién es cada jugador: están lejos, se mueven rápido y muchos se parecen con la misma equipación. Este programa intenta ayudar a esos aficionados: **reconoce la cara del jugador en el vídeo y le pone al lado una tarjeta con sus datos** (dorsal, nombre, posición, nacionalidad y edad), como una "nube de información".

## 🔍 ¿Qué hace?

El programa trabaja en dos fases:

| Fase | Qué pasa | Qué haces tú |
|---|---|---|
| **1. Registro** | Busca las caras en la foto del equipo y te las enseña una a una | Escribes el dorsal del jugador que ves (o ENTER para saltarla) |
| **2. Vídeo** | Reproduce el partido, busca caras en cada fotograma, decide a qué jugador se parece más y dibuja su tarjeta | Mirar. Pulsa `q` para salir |

En pocas palabras: **primero le enseñamos las caras al programa y luego él las busca en el vídeo.**

## 📁 Estructura del proyecto

```
Proyecto OpenCV/
├── haarcascades/
│   └── haarcascade_frontalface_default.xml   # Detector de caras
├── imagenes/
│   ├── caras_proyecto/        # Primeros planos de los jugadores de pruebas
│   ├── fotos_equipos/         # Fotos de registro
│   └── caras/          # Caras que guarda el programa al registrar (se crea sola)
├── utils/                     # Datos de los jugadores en JSON
├── videos/                    # Vídeos de prueba (barcelona.mp4, partido.mp4...)
└── proyecto_opencv_version_1.py        # Programa principal
```

## ⚙️ Requisitos e instalación

- Python 3
- OpenCV

```bash
pip install opencv-python
```

## ▶️ Ejecución

1. Abre una terminal **dentro de la carpeta del proyecto** (las rutas son relativas a ella).
2. Ejecuta:
   ```bash
   python proyecto_opencv_version_1.py
   ```
3. Registra a los jugadores escribiendo su dorsal en la consola.
4. Mira el vídeo con las tarjetas. Pulsa `q` para cerrar.

## 📊 Datos de los jugadores (JSON)

Los archivos de la carpeta `utils` guardan la información de cada equipo que aparece en las tarjetas de los jugadores. La clave es el dorsal. Por ejemplo, un jugador del archivo `utils/barcelona.json` se registra en el JSON así:

```json
{
  "11": {
    "nombre": "Raphinha",
    "posicion": "Delantero",
    "nacionalidad": "Brasil",
    "edad": 29,
    "nota": "Capitán"
  }
}
```
Como se ve, se puede añadir un campo opcional `"nota"`, que aparece como una línea extra destacada en la tarjeta.

## 🧰 Funciones de OpenCV usadas

| Función | Para qué la usamos | Explicado fácil |
|---|---|---|
| `cv2.imread` | Abrir la foto del equipo | Carga una imagen del disco |
| `cv2.imwrite` | Guardar cada cara registrada | Guarda una imagen en el disco |
| `cv2.cvtColor(..., COLOR_BGR2GRAY)` | Pasar a blanco y negro | El detector y la comparación trabajan en grises |
| `cv2.CascadeClassifier` | Cargar el detector de caras | Lee el archivo XML que "sabe" cómo es una cara de frente |
| `detectMultiScale` | Encontrar caras | Devuelve un rectángulo por cada cara encontrada |
| `cv2.resize` | Poner todas las caras a 100×100 y reducir el vídeo | Para comparar caras del mismo tamaño e ir más rápido |
| `cv2.equalizeHist` | Igualar el contraste de la cara | Reparte los grises para que una foto oscura y una clara se parezcan más |
| `cv2.matchTemplate` (`TM_SQDIFF_NORMED`) | Comparar la cara del vídeo con las guardadas | Resta píxel a píxel: cuanto más cerca de 0, más se parecen |
| `cv2.minMaxLoc` | Leer el resultado de la comparación | Saca el valor de parecido |
| `cv2.VideoCapture` / `read` / `release` | Leer el vídeo | Abre el vídeo y lo recorre fotograma a fotograma |
| `cv2.imshow` / `waitKey` / `destroyAllWindows` | Mostrar ventanas | Enseña las imágenes y espera teclas |
| `cv2.rectangle` | Marco de la cara y fondo de la tarjeta | Dibuja rectángulos (rellenos o solo borde) |
| `cv2.putText` | Escribir los datos y el valor de parecido | Escribe texto sobre la imagen |

## ⚙️ Ajustes que puedes tocar

| Variable | Valor | Qué controla |
|---|---|---|
| `archivo_video` | `./videos/barcelona.mp4` | Vídeo que se analiza |
| `archivo_foto_equipo` | `./imagenes/fotos_equipos/dorsales.webp` | Foto de la que se sacan las caras para el registro |
| `UMBRAL` | `0.3` | Diferencia máxima para aceptar un nombre. Si se supera, sale "Desconocido" |
| `escala` | `0.5` | Cuánto se reduce el vídeo antes de buscar caras (más pequeño = más rápido, pero pierde caras pequeñas) |
| `minNeighbors` | `5` | Lo exigente que es el detector (más alto = menos caras falsas) |

**Truco para ajustar el `UMBRAL`:** debajo de cada cara aparece un número amarillo, que es la diferencia con la plantilla más parecida. Fíjate en los valores cuando acierta y pon el umbral un poco por encima.

## 📖 Ejemplos usados como guía

| Ejemplo | Qué hemos aprovechado |
|---|---|
| `deteccion_caras.py` | Detección de caras con Haar |
| `identificacion_personas.py` | Comparar caras con `matchTemplate` |
| `reproducir_videos.py` | Leer y mostrar el vídeo |
| `escalado.py` | Reducir la imagen con `resize` |
| `histograma_gris.py` | Trabajar el histograma de grises (base de `equalizeHist`) |

## ⚠️ Limitaciones

Somos sinceros: el reconocimiento **no acierta siempre**. Estas son las causas:

| Limitación | ¿Por qué? |
|---|---|
| **Confunde jugadores** | `matchTemplate` no sabe qué es un ojo o una nariz: solo compara píxeles claros y oscuros. Dos caras con iluminación parecida pueden "parecerse" más que la misma persona con otra luz. El `equalizeHist` lo mejora, pero no lo elimina |
| **No detecta caras en planos abiertos** | En un plano general la cara ocupa muy pocos píxeles y el detector no la encuentra |
| **No detecta caras de perfil** | El detector Haar que usamos solo está entrenado con caras de frente |

Los sistemas de reconocimiento facial actuales resuelven esto con **redes neuronales**, que en lugar de comparar píxeles aprenden los rasgos de cada cara. Eso queda fuera de la visión por computador clásica que trabajamos en la asignatura, pero sería el siguiente paso natural.

## 👥 Autores

- **David Muñoz Torrero Pérez**
- **Miguel Alejandro García Sevilla**