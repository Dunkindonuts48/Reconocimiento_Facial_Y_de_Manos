"""Reconocimiento de expresiones faciales."""

import math


class ExpressionRecognizer:
    """Analiza los puntos faciales para reconocer expresiones."""

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

        Devuelve:

        feliz
        triste
        default
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

        # -------------------------------------------------
        # POSICIÓN DE LAS COMISURAS
        # -------------------------------------------------

        mouth_center_y = (
            mouth_top.y
            + mouth_bottom.y
        ) / 2

        corners_center_y = (
            mouth_left.y
            + mouth_right.y
        ) / 2

        # Normalizamos respecto a la anchura de la boca.
        corner_difference = (
            corners_center_y
            - mouth_center_y
        ) / mouth_width

        # -------------------------------------------------
        # FELIZ
        # -------------------------------------------------

        # Boca abierta + comisuras hacia arriba.
        mouth_ratio = (
            mouth_height
            / mouth_width
        )

        if (
            mouth_ratio > 0.20
            and corner_difference < -0.03
        ):
            return "feliz"

        # Una sonrisa más cerrada.
        if corner_difference < -0.08:
            return "feliz"

        # -------------------------------------------------
        # TRISTE
        # -------------------------------------------------

        # En una expresión triste las comisuras
        # tienden a caer respecto al centro de la boca.
        if corner_difference > 0.08:
            return "triste"

        # -------------------------------------------------
        # DEFAULT
        # -------------------------------------------------

        return "default"