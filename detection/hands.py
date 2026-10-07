"""Detección de manos utilizando MediaPipe Hand Landmarker."""

import cv2
import mediapipe as mp


class HandDetector:
    """Detector de manos basado en MediaPipe Hand Landmarker."""

    def __init__(self, model_path):
        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_path
        )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.detector = (
            mp.tasks.vision.HandLandmarker.create_from_options(
                options
            )
        )

    def process(self, frame, timestamp_ms):
        """Analiza un frame y devuelve las manos detectadas."""

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        return self.detector.detect_for_video(
            mp_image,
            timestamp_ms,
        )

    def draw(self, frame, results):
        """Dibuja los puntos de las manos detectadas."""

        if not results.hand_landmarks:
            return frame

        height, width, _ = frame.shape

        for hand_landmarks in results.hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (255, 0, 0),
                    -1,
                )

        return frame

    def close(self):
        """Libera los recursos de MediaPipe."""

        self.detector.close()