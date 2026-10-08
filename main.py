"""Programa principal del sistema de reconocimiento facial y de manos."""

import sys
import time
from pathlib import Path

import cv2

from detection.face import FaceDetector
from detection.hands import HandDetector

from recognition.expressions import ExpressionRecognizer
from recognition.gestures import GestureRecognizer
from recognition.stability import IdentifierStabilizer

from image_manager.image_manager import ImageManager
from effects.face_filter import FaceModelFilter


def create_face_adjustment_controls():
    """Crea sliders acumulativos para ajustar el modelo sin topes prácticos."""
    name = "Ajuste 3D (sliders acumulativos)"
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    controls = (
        ("Mover X (+/- px)", 500, 1000),
        ("Mover Y (+/- px)", 500, 1000),
        ("Escala (+/-)", 500, 1000),
        ("Girar yaw (+/- grados)", 500, 1000),
        ("Girar pitch (+/- grados)", 500, 1000),
        ("Girar roll (+/- grados)", 500, 1000),
        ("Invertir yaw", 1, 1),
    )
    for label, initial, maximum in controls:
        cv2.createTrackbar(label, name, initial, maximum, lambda _value: None)
    return name


def read_face_adjustments(window_name, state):
    """Acumula el desplazamiento y recentra los sliders al acercarse al borde."""
    sliders = {
        "x": ("Mover X (+/- px)", 1.0),
        "y": ("Mover Y (+/- px)", 1.0),
        "yaw": ("Girar yaw (+/- grados)", 1.0),
        "pitch": ("Girar pitch (+/- grados)", 1.0),
        "roll": ("Girar roll (+/- grados)", 1.0),
    }
    for key, (label, unit_per_tick) in sliders.items():
        position = cv2.getTrackbarPos(label, window_name)
        delta = position - state["last"][key]
        state[key] += delta * unit_per_tick
        if position < 100 or position > 900:
            cv2.setTrackbarPos(label, window_name, 500)
            state["last"][key] = 500
        else:
            state["last"][key] = position

    scale_position = cv2.getTrackbarPos("Escala (+/-)", window_name)
    scale_delta = scale_position - state["last"]["scale"]
    state["scale"] *= 1.005 ** scale_delta
    state["scale"] = min(max(state["scale"], 0.01), 1_000_000.0)
    if scale_position < 100 or scale_position > 900:
        cv2.setTrackbarPos("Escala (+/-)", window_name, 500)
        state["last"]["scale"] = 500
    else:
        state["last"]["scale"] = scale_position

    state["invert_yaw"] = bool(
        cv2.getTrackbarPos("Invertir yaw", window_name)
    )
    return {
        "x": state["x"], "y": state["y"],
        "scale": state["scale"], "yaw": state["yaw"],
        "pitch": state["pitch"], "roll": state["roll"],
        "invert_yaw": state["invert_yaw"],
    }


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

    face_3d_model_path = (
        project_dir
        / "models"
        / "barack_obama.glb"
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

    if not face_3d_model_path.exists():
        print(
            f"No se encuentra el modelo 3D facial: "
            f"{face_3d_model_path}"
        )
        sys.exit(1)

    # Comprobar carpeta de imágenes
    if not images_directory.exists():
        print(
            f"No se encuentra la carpeta de imágenes: "
            f"{images_directory}"
        )
        sys.exit(1)

    # Cargar el GLB y preparar el renderizador antes de abrir la cámara.
    face_model_filter = FaceModelFilter(face_3d_model_path)

    # Abrir cámara
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        face_model_filter.close()
        print(
            "No se ha podido abrir la cámara. "
            "Comprueba que existe y que no está siendo "
            "usada por otra aplicación."
        )
        sys.exit(1)

    face_detector = None
    hand_detector = None

    try:
        # Crear detectores dentro del bloque protegido para liberar
        # la cámara si algún modelo no se puede inicializar.
        face_detector = FaceDetector(str(face_model_path))
        hand_detector = HandDetector(str(hand_model_path))
    except Exception:
        if face_detector is not None:
            face_detector.close()
        cap.release()
        cv2.destroyAllWindows()
        raise

    # Crear reconocedor de expresiones
    expression_recognizer = ExpressionRecognizer()

    # Crear reconocedor de gestos
    gesture_recognizer = GestureRecognizer()

    # Evitar que el identificador cambie por detecciones aisladas.
    identifier_stabilizer = IdentifierStabilizer(
        confirmation_frames=5,
        hold_frames=8,
    )

    # Crear gestor de imágenes
    image_manager = ImageManager(
        images_directory
    )
    effect_mode = True
    adjustment_window = create_face_adjustment_controls()
    adjustment_state = {
        # Calibración base indicada por el usuario en la captura.
        "x": 3.0, "y": -316.0, "scale": 125.0,
        "yaw": -235.0, "pitch": 3.0, "roll": -3.0,
        "invert_yaw": True,
        "last": {"x": 500, "y": 500, "scale": 500,
                 "yaw": 500, "pitch": 500, "roll": 500},
    }

    print("Cámara abierta.")
    print("MediaPipe Face Landmarker activado.")
    print("MediaPipe Hand Landmarker activado.")
    print("Reconocimiento de expresiones activado.")
    print("Reconocimiento de gestos activado.")
    print("Gestor de imágenes activado.")
    print("Modelo 3D facial activado. Pulsa 'm' para cambiar de modo y 'q' para salir.")

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

            # Ambos detectores procesan el frame original.
            analysis_frame = frame

            # -------------------------------------------------
            # DETECCIÓN DE CARA
            # -------------------------------------------------

            face_results = face_detector.process(
                analysis_frame,
                timestamp_ms,
            )

            # Reconocer expresión
            expression = (
                expression_recognizer.analyze(
                    face_results
                )
            )

            # -------------------------------------------------
            # DETECCIÓN DE MANOS
            # -------------------------------------------------

            hand_results = hand_detector.process(
                analysis_frame,
                timestamp_ms,
            )

            # Reconocer gesto
            gesture = gesture_recognizer.analyze(
                hand_results
            )

            # -------------------------------------------------
            # DETERMINAR IDENTIFICADOR FINAL
            # -------------------------------------------------

            if gesture is not None:
                detected_identifier = gesture
            else:
                detected_identifier = expression

            identifier = identifier_stabilizer.update(
                detected_identifier
            )

            if effect_mode:
                face_adjustments = read_face_adjustments(
                    adjustment_window, adjustment_state
                )
                frame = face_model_filter.draw(
                    frame,
                    face_results,
                    hand_results,
                    face_adjustments,
                )
            else:
                frame = face_detector.draw(frame, face_results)
                frame = hand_detector.draw(frame, hand_results)

            # -------------------------------------------------
            # OBTENER IMAGEN
            # -------------------------------------------------

            result_image = None
            if not effect_mode:
                result_image = image_manager.get_image(identifier)

            # Si no encontramos imagen, utilizar default
            if not effect_mode and result_image is None:
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
                f"ID: {identifier} | Modo: {'Modelo 3D' if effect_mode else 'Imagenes'}",
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            if effect_mode:
                cv2.putText(
                    frame,
                    face_model_filter.status,
                    (30, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    (f"Ajuste X:{face_adjustments['x']:+.0f}px "
                     f"Y:{face_adjustments['y']:+.0f}px "
                     f"Escala:{face_adjustments['scale']:.0f}%"),
                    (30, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1,
                )
                cv2.putText(
                    frame,
                    (f"Yaw:{face_adjustments['yaw']:+.0f} "
                     f"Pitch:{face_adjustments['pitch']:+.0f} "
                     f"Roll:{face_adjustments['roll']:+.0f} "
                     f"Invertido:{int(face_adjustments['invert_yaw'])}"),
                    (30, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1,
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
                "q salir | m cambiar modo",
                combined,
            )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                effect_mode = not effect_mode

    finally:
        face_detector.close()
        hand_detector.close()
        face_model_filter.close()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
