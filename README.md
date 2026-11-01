# ⚽ Reconocimiento de jugadores — Versión 2 (con librerías)

Segunda versión del proyecto de **Imagen Digital** (Ingeniería del Software, UEx).

El objetivo es el mismo que en la versión 1: ayudar a los aficionados que no distinguen a los jugadores mostrando una **tarjeta de información** junto a cada jugador en el vídeo del partido.

La diferencia es **cómo se reconoce a cada jugador**: la versión 1 usa solo lo visto en clase con OpenCV, y esta versión 2 usa además la librería **`face_recognition`** (basada en redes neuronales). La presentamos como **opción alternativa** porque los resultados son mucho mejores.

---

## 📁 Archivo

| Archivo | Descripción |
|---|---|
| `Entrega_Poryecto_OpenCV_Versión2.py` | Programa completo de la versión 2 |
| `utils/barcelona.json` | Datos de los jugadores (nombre, posición, nacionalidad, edad, nota) |
| `imagenes/caras/dorsales.webp` | Foto de registro de los jugadores |
| `imagenes/caras_prueba/` | Caras que se guardan al registrar |
| `videos/barcelona.mp4` | Vídeo de la demo |

---

## ⚙️ Instalación

```bash
pip install "opencv-python<5" face_recognition
python -m pip install "setuptools<70.0.0" --force-reinstall
python -m pip install --force-reinstall face_recognition face-recognition-models cmake dlib
```

| Nota | Detalle |
|---|---|
| `opencv-python<5` | Los códigos de clase están hechos para OpenCV 4 |
| `dlib` en Windows | `face_recognition` depende de `dlib`; puede necesitar las herramientas de C++ de Visual Studio |
| `setuptools<70` | Evita un error de compatibilidad al cargar los modelos de `face_recognition` |

## ▶️ Ejecución

```bash
python Entrega_Poryecto_OpenCV_Versión2.py
```

1. Aparece cada cara de la foto de registro y se escribe por consola el **dorsal** del jugador (ENTER para ignorar).
2. Se reproduce el vídeo con el recuadro y la tarjeta de cada jugador reconocido.
3. La **barra "Tiempo"** permite avanzar o retroceder en el vídeo. Tecla **`q`** para salir.

---

## 🔄 Versión 1 vs Versión 2

| Aspecto | Versión 1 (solo OpenCV) | Versión 2 (con `face_recognition`) |
|---|---|---|
| Detectar caras | Haar Cascade (`CascadeClassifier`) | Detector HOG (`face_locations`) |
| Reconocer quién es | Compara píxeles (`matchTemplate`) | Compara 128 rasgos de la cara (`face_encodings`) |
| Color | Gris + `equalizeHist` | Color (RGB) |
| Fiabilidad | Confunde jugadores con frecuencia | Acierta en la gran mayoría de casos |
| Desconocidos | Umbral de diferencia de píxeles (0.3) | Distancia entre rasgos menor de 0.6 |
| Velocidad | Fluida | Más pesada → optimizada (ver abajo) |
| Barra de tiempo | ❌ | ✅ |

---

## 🛠️ Cambios realizados (explicados por encima)

### 1. Reconocimiento con red neuronal
En lugar de comparar píxel a píxel (que falla en cuanto cambia la luz o el ángulo), la librería convierte cada cara en **128 números** que describen sus rasgos (distancia entre ojos, forma de la nariz, mandíbula…). Dos caras de la misma persona dan números muy parecidos aunque la foto sea distinta.

| Función | Qué hace |
|---|---|
| `face_recognition.face_locations()` | Encuentra las caras en la imagen |
| `face_recognition.face_encodings()` | Saca los 128 rasgos de cada cara ("huella facial") |
| `face_recognition.face_distance()` | Mide cuánto se parecen dos huellas (0 = idénticas) |
| `np.argmin()` | Elige al jugador registrado más parecido |

Si la distancia más pequeña es **menor de 0.6**, se acepta el nombre; si no, se marca como **"Desconocido"** en rojo.

### 2. Registro guiado con el JSON
Al registrar se muestra la lista de jugadores del `barcelona.json` y se escribe el **dorsal**. El programa valida la entrada y quita de la lista a los ya asignados, para no repetir.

### 3. Optimización de velocidad
La IA es mucho más pesada que Haar, así que el vídeo iba muy lento. Cambios aplicados:

| Cambio | Qué soluciona |
|---|---|
| **Barra que no "salta" sola** | Al actualizar la barra, OpenCV volvía a colocar el vídeo en cada fotograma, lo que era muy lento. Ahora solo salta cuando el usuario la mueve con el ratón |
| **La IA analiza 1 de cada 3 fotogramas** (`CADA_N_FRAMES`) | En los fotogramas intermedios se reutilizan los últimos recuadros |
| **Imagen reducida a 480 px de ancho** (`ANCHO_IA`) | La IA trabaja con una imagen pequeña, sea cual sea la resolución del vídeo |
| **La barra se actualiza cada 10 fotogramas** (`CADA_N_BARRA`) | Menos trabajo de dibujo |

Estos tres valores están al principio del código para ajustarlos según el ordenador.

### 4. Barra de tiempo
Con `cv2.createTrackbar` se añade una barra para moverse por el vídeo, útil en la demo para ir directamente a las jugadas que interesan.

### 5. Tarjeta de información
Junto a cada jugador reconocido se dibuja una tarjeta con dorsal, nombre, posición, nacionalidad, edad y nota, leídos del JSON.

---

## 🧰 Funciones de OpenCV usadas

| Función | Para qué |
|---|---|
| `cv2.imread` | Cargar la foto de registro |
| `cv2.cvtColor(..., COLOR_BGR2RGB)` | OpenCV trabaja en BGR y `face_recognition` en RGB |
| `cv2.resize` | Ampliar la cara al registrar y reducir el frame para la IA |
| `cv2.imshow` / `cv2.waitKey` / `cv2.destroyWindow` | Mostrar ventanas y leer teclas |
| `cv2.imwrite` | Guardar las caras registradas |
| `cv2.VideoCapture` / `read` / `release` | Abrir y leer el vídeo |
| `cv2.CAP_PROP_FRAME_COUNT` / `CAP_PROP_POS_FRAMES` | Total de fotogramas y posición actual |
| `cv2.namedWindow` / `createTrackbar` / `setTrackbarPos` | Barra de tiempo |
| `cv2.rectangle` | Recuadro de la cara y fondo de la tarjeta |
| `cv2.putText` | Nombre y datos de la tarjeta |

---

## ⚠️ Limitaciones

| Limitación | Por qué |
|---|---|
| Sigue habiendo algo de lag | `face_recognition` calcula con el procesador; con una tarjeta gráfica NVIDIA iría fluido, pero requiere compilar `dlib` con CUDA |
| Planos abiertos | Si la cara ocupa muy pocos píxeles, no se detecta |
| Tildes y ñ salen como `??` | `cv2.putText` no admite esos caracteres |
| Usa librerías externas | Por eso la versión 1 (solo OpenCV) sigue siendo la versión principal |

---

## 👥 Autores

- David Muñoz Torrero Pérez
- Miguel Alejandro García Sevilla