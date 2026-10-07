"""Detección facial utilizando MediaPipe Face Landmarker."""

import cv2
import mediapipe as mp


class FaceDetector:
    """Detector de caras basado en MediaPipe Face Landmarker."""

    def __init__(self, model_path):
        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_path
        )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.detector = (
            mp.tasks.vision.FaceLandmarker.create_from_options(options)
        )

    def process(self, frame, timestamp_ms):
        """Analiza un frame y devuelve los puntos faciales."""

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        return self.detector.detect_for_video(
            mp_image,
            timestamp_ms,
        )

    def draw(self, frame, results):
        """Dibuja los puntos detectados sobre la cara."""

        if not results.face_landmarks:
            return frame

        height, width, _ = frame.shape

        for face_landmarks in results.face_landmarks:
            for landmark in face_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1,
                )

        return frame

    def close(self):
        """Libera los recursos de MediaPipe."""

        self.detector.close()