"""Reconocimiento de expresiones faciales."""

import math


class ExpressionRecognizer:
    """Analiza los puntos faciales para reconocer expresiones."""

    # Puntos principales de la boca en MediaPipe Face Landmarker
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14

    def __init__(self):
        pass

    @staticmethod
    def _distance(point_a, point_b):
        """Calcula la distancia entre dos puntos faciales."""

        return math.sqrt(
            (point_a.x - point_b.x) ** 2
            + (point_a.y - point_b.y) ** 2
        )

    def analyze(self, results):
        """
        Analiza una detección facial.

        Devuelve un identificador de expresión.
        """

        if not results.face_landmarks:
            return "default"

        face = results.face_landmarks[0]

        mouth_left = face[self.MOUTH_LEFT]
        mouth_right = face[self.MOUTH_RIGHT]
        mouth_top = face[self.MOUTH_TOP]
        mouth_bottom = face[self.MOUTH_BOTTOM]

        # Anchura de la boca
        mouth_width = self._distance(
            mouth_left,
            mouth_right,
        )

        # Altura de la boca
        mouth_height = self._distance(
            mouth_top,
            mouth_bottom,
        )

        if mouth_width == 0:
            return "default"

        # Relación entre anchura y altura.
        mouth_ratio = mouth_height / mouth_width

        # Primera regla experimental.
        #
        # Una boca suficientemente abierta y ancha
        # puede corresponder a una sonrisa.
        if mouth_ratio > 0.20:
            return "feliz"

        return "default"