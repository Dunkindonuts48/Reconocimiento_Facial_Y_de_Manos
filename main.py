"""Programa principal del sistema de reconocimiento facial y de manos."""

import sys
import time
from pathlib import Path

import cv2

from detection.face import FaceDetector
from detection.hands import HandDetector

from recognition.expressions import ExpressionRecognizer
from recognition.gestures import GestureRecognizer

from image_manager.image_manager import ImageManager


def main():
    # Directorio principal del proyecto
    project_dir = Path(__file__).parent

    # Ruta del modelo facial
    face_model_path = (
        project_dir
        / "models"
        / "face_landmarker.task"
    )

    # Ruta del modelo de manos
    hand_model_path = (
        project_dir
        / "models"
        / "hand_landmarker.task"
    )

    # Ruta de las imágenes
    images_directory = (
        project_dir
        / "Imagenes"
    )

    # Comprobar modelo facial
    if not face_model_path.exists():
        print(
            f"No se encuentra el modelo facial: "
            f"{face_model_path}"
        )
        sys.exit(1)

    # Comprobar modelo de manos
    if not hand_model_path.exists():
        print(
            f"No se encuentra el modelo de manos: "
            f"{hand_model_path}"
        )
        sys.exit(1)

    # Comprobar carpeta de imágenes
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
            "Comprueba que existe y que no está siendo "
            "usada por otra aplicación."
        )
        sys.exit(1)

    # Crear detector facial
    face_detector = FaceDetector(
        str(face_model_path)
    )

    # Crear detector de manos
    hand_detector = HandDetector(
        str(hand_model_path)
    )

    # Crear reconocedor de expresiones
    expression_recognizer = ExpressionRecognizer()

    # Crear reconocedor de gestos
    gesture_recognizer = GestureRecognizer()

    # Crear gestor de imágenes
    image_manager = ImageManager(
        images_directory
    )

    print("Cámara abierta.")
    print("MediaPipe Face Landmarker activado.")
    print("MediaPipe Hand Landmarker activado.")
    print("Reconocimiento de expresiones activado.")
    print("Reconocimiento de gestos activado.")
    print("Gestor de imágenes activado.")
    print("Pulsa 'q' para salir.")

    start_time = time.monotonic()

    try:
        while True:
            # Leer frame de la cámara
            ret, frame = cap.read()

            if not ret:
                print(
                    "Error al leer frame de la cámara."
                )
                break

            # Tiempo transcurrido en milisegundos
            timestamp_ms = int(
                (time.monotonic() - start_time)
                * 1000
            )

            # -------------------------------------------------
            # DETECCIÓN DE CARA
            # -------------------------------------------------

            face_results = face_detector.process(
                frame,
                timestamp_ms,
            )

            # Reconocer expresión
            expression = (
                expression_recognizer.analyze(
                    face_results
                )
            )

            # Dibujar puntos faciales
            frame = face_detector.draw(
                frame,
                face_results,
            )

            # -------------------------------------------------
            # DETECCIÓN DE MANOS
            # -------------------------------------------------

            hand_results = hand_detector.process(
                frame,
                timestamp_ms,
            )

            # Reconocer gesto
            gesture = gesture_recognizer.analyze(
                hand_results
            )

            # Dibujar puntos de las manos
            frame = hand_detector.draw(
                frame,
                hand_results,
            )

            # -------------------------------------------------
            # DETERMINAR IDENTIFICADOR FINAL
            # -------------------------------------------------

            if gesture is not None:
                identifier = gesture
            else:
                identifier = expression

            # -------------------------------------------------
            # OBTENER IMAGEN
            # -------------------------------------------------

            result_image = (
                image_manager.get_image(
                    identifier
                )
            )

            # Si no encontramos imagen, utilizar default
            if result_image is None:
                result_image = (
                    image_manager.get_image(
                        "default"
                    )
                )

            # -------------------------------------------------
            # INFORMACIÓN SOBRE LA CÁMARA
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"Expresion: {expression}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            if gesture is not None:
                gesture_text = (
                    f"Gesto: {gesture}"
                )
            else:
                gesture_text = "Gesto: ninguno"

            cv2.putText(
                frame,
                gesture_text,
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )

            cv2.putText(
                frame,
                f"ID: {identifier}",
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            # -------------------------------------------------
            # REDIMENSIONAR IMAGEN DE RESULTADO
            # -------------------------------------------------

            if result_image is not None:

                # Copia para no modificar la imagen
                # almacenada en ImageManager
                result_image = (
                    result_image.copy()
                )

                # Altura de la cámara
                camera_height = frame.shape[0]

                # Tamaño original
                result_height, result_width = (
                    result_image.shape[:2]
                )

                # Mantener proporción
                new_width = int(
                    result_width
                    * camera_height
                    / result_height
                )

                result_image = cv2.resize(
                    result_image,
                    (
                        new_width,
                        camera_height,
                    ),
                )

                # Unir cámara + imagen
                combined = cv2.hconcat(
                    [
                        frame,
                        result_image,
                    ]
                )

            else:
                combined = frame

            # -------------------------------------------------
            # MOSTRAR RESULTADO
            # -------------------------------------------------

            cv2.imshow(
                "Reconocimiento facial y de manos - "
                "Pulsa q para salir",
                combined,
            )

            # Salir con Q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        face_detector.close()
        hand_detector.close()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()