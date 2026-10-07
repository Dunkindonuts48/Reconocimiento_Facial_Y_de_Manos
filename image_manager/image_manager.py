"""Gestión de las imágenes asociadas a cada identificador."""

from pathlib import Path

import cv2


class ImageManager:
    """Relaciona identificadores con imágenes."""

    IMAGE_MAP = {
        "feliz": "Gato_Feliz.jpg",
        "triste": "Gato_triste.jpg",
        "dedo_arriba": "Gato_dedo_arriba.jpg",
        "pulgar_arriba": "Gato_dedo_pulgar_arriba.jpg",
        "absolut_cinema": "Gato_absolut_cinema.png",
        "default": "Gato_default.jpg",
    }

    def __init__(self, images_directory):
        self.images_directory = Path(images_directory)
        self.images = {}

        self._load_images()

    def _load_images(self):
        """Carga todas las imágenes en memoria."""

        for identifier, filename in self.IMAGE_MAP.items():
            image_path = self.images_directory / filename

            if not image_path.exists():
                print(
                    f"Aviso: no existe la imagen: {image_path}"
                )
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                print(
                    f"Aviso: no se ha podido cargar: {image_path}"
                )
                continue

            self.images[identifier] = image

    def get_image(self, identifier):
        """Devuelve la imagen asociada al identificador."""

        return self.images.get(
            identifier,
            self.images.get("default"),
        )