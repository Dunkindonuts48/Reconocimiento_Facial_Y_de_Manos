# Reconocimiento facial y de manos

Aplicación de escritorio en Python que analiza en tiempo real la imagen de una cámara, reconoce algunas expresiones y gestos, y muestra una imagen asociada al identificador detectado.

## Estado del proyecto

La aplicación integra captura de cámara, detección facial y de manos con MediaPipe, reconocimiento heurístico y visualización de resultados. Es un prototipo: las expresiones se estiman a partir de la forma de la boca y los gestos mediante reglas sobre los puntos de la mano. No se utiliza un modelo entrenado para clasificar emociones.

### Funciones implementadas

- Captura de vídeo desde la cámara predeterminada.
- Detección y dibujo de puntos faciales y de manos.
- Expresiones: feliz, triste y estado predeterminado.
- Gestos: dedo arriba, pulgar arriba y dos manos levantadas.
- Asociación de identificadores con imágenes locales.
- Estabilización temporal para reducir cambios por detecciones aisladas.
- Modelo facial 3D texturizado desde `models/barack_obama.glb`, ajustado a la pose detectada y renderizado con OpenGL.
- Repulsores luminosos sobre las palmas.
- Modo de imágenes original, que se puede recuperar pulsando **m**.

El filtro se activa por defecto. Pulsa **m** para alternar entre el filtro y el modo de imágenes; pulsa **q** para cerrar. La cámara virtual para Microsoft Teams/OBS sigue pendiente.

Al usar el modelo 3D aparece la ventana **Ajuste 3D (sliders acumulativos)**. Los controles de desplazamiento y giro se recentran al llegar cerca del borde y siguen acumulando el ajuste; la escala aumenta o disminuye de forma multiplicativa. Los valores acumulados se muestran sobre la cámara. **Invertir yaw** cambia el sentido del giro horizontal.

## Requisitos e instalación

- Python 3.10 o una versión compatible con las dependencias instaladas.
- Una cámara disponible para OpenCV.
- Los modelos de MediaPipe incluidos en la carpeta models.

Desde la raíz del repositorio, crea el entorno e instala las dependencias:

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

Si PowerShell bloquea la activación, ejecuta el intérprete del entorno directamente:

    .venv\Scripts\python.exe main.py

## Ejecución

Desde la raíz del proyecto:

    python main.py

Pulsa **q** para cerrar la ventana. Si no se abre la cámara, comprueba que otra aplicación no la esté usando y que Windows haya concedido permiso de cámara a Python.

## Estructura

    main.py                         Coordinación de cámara, detección y ventana
    detection/face.py               MediaPipe Face Landmarker
    detection/hands.py              MediaPipe Hand Landmarker
    recognition/expressions.py      Reglas heurísticas para expresiones
    recognition/gestures.py         Reglas para gestos de manos
    recognition/stability.py        Persistencia temporal de identificadores
    effects/glb_face_model.py       Lectura y recorte de la malla GLB
    effects/face_filter.py          Renderizado OpenGL de la cara y efectos de manos
    models/barack_obama.glb         Modelo 3D facial y textura
    image_manager/image_manager.py  Carga y asociación de imágenes
    models/                         Modelos .task de MediaPipe
    Imagenes/                       Imágenes mostradas por identificador
    requirements.txt                Dependencias de Python

## Identificadores e imágenes

El mapa de asociaciones está en ImageManager.IMAGE_MAP, dentro de image_manager/image_manager.py. Para añadir una imagen:

1. Copia el archivo a la carpeta Imagenes.
2. Añade el identificador y el nombre del archivo al mapa.
3. Devuelve ese identificador desde el reconocedor correspondiente.

Si un archivo no existe o no se puede cargar, el gestor avisa en la consola y utiliza la imagen predeterminada cuando está disponible.

## Estabilidad del reconocimiento

IdentifierStabilizer exige que un identificador nuevo se detecte durante cinco frames consecutivos antes de mostrarlo. Si la detección desaparece brevemente, conserva el identificador actual y vuelve al estado predeterminado tras ocho frames sin una detección válida. Los valores se configuran en main.py.

## Próximas mejoras

- Hacer configurables el índice de cámara, la resolución y los umbrales.
- Mejorar el reconocimiento de gestos teniendo en cuenta la orientación y la posición de la persona.
- Evaluar las categorías de expresiones de MediaPipe en distintas condiciones de luz y con distintas cámaras.
- Separar la interfaz de la lógica de captura y reconocimiento.
- Añadir una salida de cámara virtual para OBS/Teams.

## Limitaciones conocidas

- Las reglas geométricas son sensibles al ángulo de la mano, la distancia a la cámara y la iluminación.
- El filtro 3D requiere soporte OpenGL en el controlador gráfico de Windows.
- La expresión se estima solamente con la geometría de la boca. El estado predeterminado indica que las reglas actuales no detectan sonrisa o tristeza; no significa que se haya reconocido una emoción neutral.
- Un gesto de mano tiene prioridad sobre una expresión facial.
- La cámara utiliza el índice 0; todavía no hay selector de dispositivo.

## Atribución del modelo 3D

El modelo BARACK OBAMA de LOUIS se distribuye bajo licencia CC BY 4.0. Esta aplicación usa su malla y textura para el seguimiento facial.

- Fuente: https://sketchfab.com/3d-models/barack-obama-3010d6b0fdd843b397fa9e98437bc22a
- Autor: LOUIS (https://sketchfab.com/louis)
- Licencia: https://creativecommons.org/licenses/by/4.0/

El renderizado usa OpenGL a través de ModernGL para procesar la malla y su profundidad en GPU. Instala `moderngl` y `glcontext` desde `requirements.txt`.
