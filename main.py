"""Programa principal del sistema de reconocimiento facial."""

import sys
import time
from pathlib import Path

import cv2

from detection.face import FaceDetector
from recognition.expressions import ExpressionRecognizer
from image_manager.image_manager import ImageManager


def main():
    # Directorio principal del proyecto
    project_dir = Path(__file__).parent

    # Ruta del modelo de MediaPipe
    model_path = (
        project_dir
        / "models"
        / "face_landmarker.task"
    )

    if not model_path.exists():
        print(f"No se encuentra el modelo: {model_path}")
        sys.exit(1)

    # Ruta de las imágenes
    images_directory = project_dir / "Imagenes"

    if not images_directory.exists():
        print(
            f"No se encuentra la carpeta de imágenes: "
            f"{images_directory}"
        )
        sys.exit(1)

    # Abrir cámara
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(
            "No se ha podido abrir la cámara. "
            "Comprueba que existe y que no está siendo usada "
            "por otra aplicación."
        )
        sys.exit(1)

    # Crear detector facial
    face_detector = FaceDetector(str(model_path))

    # Crear reconocedor de expresiones
    expression_recognizer = ExpressionRecognizer()

    # Crear gestor de imágenes
    image_manager = ImageManager(images_directory)

    print("Cámara abierta.")
    print("MediaPipe Face Landmarker activado.")
    print("Reconocimiento de expresiones activado.")
    print("Gestor de imágenes activado.")
    print("Pulsa 'q' para salir.")

    start_time = time.monotonic()

    try:
        while True:
            # Leer frame de la cámara
            ret, frame = cap.read()

            if not ret:
                print("Error al leer frame de la cámara.")
                break

            # Tiempo transcurrido en milisegundos
            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            # Detectar cara
            results = face_detector.process(
                frame,
                timestamp_ms,
            )

            # Reconocer expresión
            expression = expression_recognizer.analyze(
                results
            )

            # Obtener imagen asociada a la expresión
            result_image = image_manager.get_image(
                expression
            )

            # Dibujar puntos faciales
            frame = face_detector.draw(
                frame,
                results,
            )

            # Mostrar expresión sobre la cámara
            cv2.putText(
                frame,
                f"Expresion: {expression}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Si no existe la imagen, utilizar la imagen por defecto
            if result_image is None:
                result_image = image_manager.get_image(
                    "default"
                )

            # Comprobar que tenemos una imagen válida
            if result_image is not None:

                # Hacer una copia para no modificar
                # la imagen almacenada en ImageManager
                result_image = result_image.copy()

                # Altura de la cámara
                camera_height = frame.shape[0]

                # Tamaño original de la imagen
                result_height, result_width = (
                    result_image.shape[:2]
                )

                # Calcular nuevo ancho manteniendo proporción
                new_width = int(
                    result_width
                    * camera_height
                    / result_height
                )

                # Redimensionar imagen
                result_image = cv2.resize(
                    result_image,
                    (new_width, camera_height),
                )

                # Unir cámara + imagen
                combined = cv2.hconcat(
                    [
                        frame,
                        result_image,
                    ]
                )

            else:
                # Si no hay imagen, mostrar solamente la cámara
                combined = frame

            # Mostrar resultado final
            cv2.imshow(
                "Reconocimiento facial - Pulsa q para salir",
                combined,
            )

            # Salir pulsando Q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        face_detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()