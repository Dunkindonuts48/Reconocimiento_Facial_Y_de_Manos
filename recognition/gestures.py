"""Reconocimiento de gestos y poses de manos."""

import math


class GestureRecognizer:
    """Reconoce gestos utilizando los puntos de MediaPipe."""

    WRIST = 0

    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4

    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8

    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12

    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16

    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    @staticmethod
    def _distance(point_a, point_b):
        """Calcula la distancia entre dos puntos."""

        return math.sqrt(
            (point_a.x - point_b.x) ** 2
            + (point_a.y - point_b.y) ** 2
        )

    @staticmethod
    def _angle(point_a, point_b, point_c):
        """
        Calcula el ángulo ABC en grados.
        """

        vector_ba = (
            point_a.x - point_b.x,
            point_a.y - point_b.y,
        )

        vector_bc = (
            point_c.x - point_b.x,
            point_c.y - point_b.y,
        )

        magnitude_ba = math.sqrt(
            vector_ba[0] ** 2
            + vector_ba[1] ** 2
        )

        magnitude_bc = math.sqrt(
            vector_bc[0] ** 2
            + vector_bc[1] ** 2
        )

        if magnitude_ba == 0 or magnitude_bc == 0:
            return 0

        dot_product = (
            vector_ba[0] * vector_bc[0]
            + vector_ba[1] * vector_bc[1]
        )

        cosine = dot_product / (
            magnitude_ba * magnitude_bc
        )

        cosine = max(-1.0, min(1.0, cosine))

        return math.degrees(
            math.acos(cosine)
        )

    def _finger_extended(
        self,
        landmarks,
        mcp_index,
        pip_index,
        dip_index,
        tip_index,
    ):
        """
        Determina si un dedo está extendido.

        Utilizamos el ángulo del dedo y la distancia
        desde la punta hasta la muñeca.
        """

        mcp = landmarks[mcp_index]
        pip = landmarks[pip_index]
        dip = landmarks[dip_index]
        tip = landmarks[tip_index]
        wrist = landmarks[self.WRIST]

        pip_angle = self._angle(
            mcp,
            pip,
            dip,
        )

        dip_angle = self._angle(
            pip,
            dip,
            tip,
        )

        tip_distance = self._distance(
            wrist,
            tip,
        )

        pip_distance = self._distance(
            wrist,
            pip,
        )

        return (
            pip_angle > 155
            and dip_angle > 155
            and tip_distance > pip_distance
        )

    def _index_extended(self, landmarks):
        """Comprueba si el índice está extendido."""

        return self._finger_extended(
            landmarks,
            self.INDEX_MCP,
            self.INDEX_PIP,
            self.INDEX_DIP,
            self.INDEX_TIP,
        )

    def _middle_extended(self, landmarks):
        """Comprueba si el dedo medio está extendido."""

        return self._finger_extended(
            landmarks,
            self.MIDDLE_MCP,
            self.MIDDLE_PIP,
            self.MIDDLE_DIP,
            self.MIDDLE_TIP,
        )

    def _ring_extended(self, landmarks):
        """Comprueba si el anular está extendido."""

        return self._finger_extended(
            landmarks,
            self.RING_MCP,
            self.RING_PIP,
            self.RING_DIP,
            self.RING_TIP,
        )

    def _pinky_extended(self, landmarks):
        """Comprueba si el meñique está extendido."""

        return self._finger_extended(
            landmarks,
            self.PINKY_MCP,
            self.PINKY_PIP,
            self.PINKY_DIP,
            self.PINKY_TIP,
        )

    def _other_fingers_closed(self, landmarks):
        """Comprueba que medio, anular y meñique estén cerrados."""

        return not (
            self._middle_extended(landmarks)
            or self._ring_extended(landmarks)
            or self._pinky_extended(landmarks)
        )

    def _is_index_gesture(self, landmarks):
        """Detecta el índice levantado."""

        return (
            self._index_extended(landmarks)
            and self._other_fingers_closed(landmarks)
        )

    def _is_thumb_gesture(self, landmarks):
        """Detecta el gesto de pulgar arriba."""

        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]
        thumb_mcp = landmarks[self.THUMB_MCP]

        # El pulgar debe subir claramente.
        thumb_is_up = (
            thumb_tip.y < thumb_ip.y
            and thumb_tip.y < thumb_mcp.y
        )

        # El índice NO puede estar extendido.
        index_is_closed = not self._index_extended(
            landmarks
        )

        other_fingers_closed = self._other_fingers_closed(
            landmarks
        )

        return (
            thumb_is_up
            and index_is_closed
            and other_fingers_closed
        )

    def _is_hand_raised(self, landmarks):
        """
        Comprueba si una mano está levantada.

        Se utiliza principalmente para detectar
        la pose absolut_cinema.
        """

        wrist = landmarks[self.WRIST]

        index_tip = landmarks[self.INDEX_TIP]
        middle_tip = landmarks[self.MIDDLE_TIP]

        return (
            wrist.y < 0.75
            and index_tip.y < 0.45
            and middle_tip.y < 0.45
        )

    def _is_absolut_cinema(self, all_hands):
        """Detecta dos manos levantadas."""

        if len(all_hands) < 2:
            return False

        raised_hands = 0

        for landmarks in all_hands:
            if self._is_hand_raised(landmarks):
                raised_hands += 1

        return raised_hands >= 2

    def analyze(self, results):
        """
        Analiza las manos y devuelve un identificador.
        """

        if not results.hand_landmarks:
            return None

        all_hands = results.hand_landmarks

        # Prioridad 1:
        # dos manos levantadas.
        if self._is_absolut_cinema(all_hands):
            return "absolut_cinema"

        # Prioridad 2:
        # índice levantado.
        for landmarks in all_hands:
            if self._is_index_gesture(landmarks):
                return "dedo_arriba"

        # Prioridad 3:
        # pulgar arriba.
        for landmarks in all_hands:
            if self._is_thumb_gesture(landmarks):
                return "pulgar_arriba"

        return None