"""Estabiliza los identificadores para evitar cambios entre frames."""


class IdentifierStabilizer:
    """Exige persistencia antes de cambiar el identificador visible."""

    def __init__(self, confirmation_frames=5, hold_frames=8, default="default"):
        if confirmation_frames < 1:
            raise ValueError("confirmation_frames debe ser al menos 1")
        if hold_frames < 0:
            raise ValueError("hold_frames no puede ser negativo")

        self.confirmation_frames = confirmation_frames
        self.hold_frames = hold_frames
        self.default = default
        self.current = default
        self._candidate = None
        self._candidate_count = 0
        self._missing_count = 0

    def update(self, identifier):
        """Actualiza la detección del frame y devuelve el ID estable."""
        if identifier is None or identifier == self.default:
            self._candidate = None
            self._candidate_count = 0
            self._missing_count += 1
            if self._missing_count > self.hold_frames:
                self.current = self.default
            return self.current

        self._missing_count = 0
        if identifier == self.current:
            self._candidate = None
            self._candidate_count = 0
            return self.current

        if identifier == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = identifier
            self._candidate_count = 1

        if self._candidate_count >= self.confirmation_frames:
            self.current = identifier
            self._candidate = None
            self._candidate_count = 0

        return self.current
