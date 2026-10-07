# Reconocimiento facial y de manos

Aplicación de escritorio en Python que analiza en tiempo real la imagen de una cámara, reconoce algunas expresiones y gestos, y muestra una imagen asociada al identificador detectado.

## Estado del proyecto

La aplicación integra captura de cámara, detección facial y de manos con MediaPipe, reconocimiento heurístico y visualización de la imagen asociada. Es un prototipo: las expresiones se estiman a partir de la forma de la boca y los gestos mediante reglas sobre los puntos de la mano. No se utiliza un modelo entrenado para clasificar emociones.

### Funciones implementadas

- Captura de vídeo desde la cámara predeterminada.
- Detección y dibujo de puntos faciales y de manos.
- Expresiones: feliz, triste y estado predeterminado.
- Gestos: dedo arriba, pulgar arriba y dos manos levantadas.
- Asociación de identificadores con imágenes locales.
- Estabilización temporal para reducir cambios por detecciones aisladas.
- Ventana de escritorio con la cámara y la imagen seleccionada.

La cámara virtual para Microsoft Teams/OBS sigue pendiente. La aplicación actual muestra la cámara en su propia ventana.

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
- La expresión se estima solamente con la geometría de la boca. El estado predeterminado indica que las reglas actuales no detectan sonrisa o tristeza; no significa que se haya reconocido una emoción neutral.
- Un gesto de mano tiene prioridad sobre una expresión facial.
- La cámara utiliza el índice 0; todavía no hay selector de dispositivo.
