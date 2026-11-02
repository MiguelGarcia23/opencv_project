# ⚽ Proyecto de reconocimiento de jugadores en vídeo

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
├── proyecto_opencv_version_1.py        # Programa principal de la versión 1 con OpenCV
└── proyecto_opencv_version_2.py        # Programa principal de la versión 2 con face_recognition
```

## ⚙️ Requisitos, instalación y ejecución

Para la ejecución de cada una de las versiones, siga el archivo `README.md`, correspondiente a la versión que se quiere ejecutar, alojado en la rama de `Git` deseada:

- `version-1`: proyecto basado únicamente en funciones de **OpenCV**
- `version-2`: proyecto que incorpora redes neuronales con el uso de la librería **face_recognition**

## 👥 Autores

- **David Muñoz Torrero Pérez**
- **Miguel Alejandro García Sevilla**