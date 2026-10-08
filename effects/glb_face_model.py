"""Carga la malla facial y la textura de un archivo GLB."""

import json
import struct
from pathlib import Path

import cv2
import numpy as np


_COMPONENTS = {
    5121: (np.dtype("u1"), 1),
    5123: (np.dtype("<u2"), 1),
    5125: (np.dtype("<u4"), 1),
    5126: (np.dtype("<f4"), 1),
}
_TYPE_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


class GLBFaceModel:
    """Modelo facial GLB con coordenadas UV y textura."""

    def __init__(self, path):
        self.path = Path(path)
        document, binary = self._read_glb(self.path)
        positions, uvs, triangles = self._collect_meshes(document, binary)
        texture = self._read_texture(document, binary)

        # El archivo es un busto. Recortamos el cuello y el torso para usar
        # solamente la cabeza sobre el rostro de la cámara.
        low_y = float(positions[:, 1].min())
        high_y = float(positions[:, 1].max())
        head_cut = low_y + (high_y - low_y) * 0.35
        keep = np.all(positions[triangles, 1] >= head_cut, axis=1)
        triangles = triangles[keep]
        if not len(triangles):
            raise ValueError("El GLB no contiene una malla facial utilizable.")

        used = np.unique(triangles)
        remap = np.full(len(positions), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        positions = positions[used]
        uvs = uvs[used]
        triangles = remap[triangles]

        bounds_min = positions.min(axis=0)
        bounds_max = positions.max(axis=0)
        center = (bounds_min + bounds_max) * 0.5
        size = np.maximum(bounds_max - bounds_min, 1e-5)

        # Coordenadas: X hacia los lados, Y hacia abajo y Z hacia la cámara.
        points = positions - center
        points[:, 0] *= 164.0 / size[0]
        points[:, 1] *= -200.0 / size[1]
        points[:, 2] *= 200.0 / size[1]

        self.vertices = points.astype(np.float32)
        self.uvs = np.clip(uvs, 0.0, 1.0).astype(np.float32)
        self.triangles = triangles.astype(np.uint32)
        self.texture = texture

    @staticmethod
    def _read_glb(path):
        data = Path(path).read_bytes()
        if len(data) < 20 or data[:4] != b"glTF":
            raise ValueError(f"No es un archivo GLB válido: {path}")
        version, total_length = struct.unpack_from("<II", data, 4)
        if version != 2 or total_length > len(data):
            raise ValueError("El archivo GLB no usa el formato 2.0.")

        offset = 12
        document = None
        binary = None
        while offset + 8 <= total_length:
            length, chunk_type = struct.unpack_from("<II", data, offset)
            offset += 8
            chunk = data[offset:offset + length]
            offset += length
            if chunk_type == 0x4E4F534A:
                document = json.loads(chunk.decode("utf-8").rstrip(" \t\r\n\x00"))
            elif chunk_type == 0x004E4942:
                binary = chunk

        if document is None or binary is None:
            raise ValueError("El GLB no contiene los bloques JSON y BIN requeridos.")
        return document, binary

    @staticmethod
    def _accessor(document, binary, accessor_index):
        accessor = document["accessors"][accessor_index]
        view = document["bufferViews"][accessor["bufferView"]]
        dtype, _ = _COMPONENTS[accessor["componentType"]]
        width = _TYPE_WIDTH[accessor["type"]]
        count = accessor["count"]
        stride = view.get("byteStride", dtype.itemsize * width)
        offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        result = np.ndarray(
            (count, width),
            dtype=dtype,
            buffer=binary,
            offset=offset,
            strides=(stride, dtype.itemsize),
        )
        return np.array(result, copy=True)

    @staticmethod
    def _node_matrix(node):
        if "matrix" in node:
            return np.asarray(node["matrix"], dtype=np.float64).reshape(
                (4, 4), order="F"
            )

        translation = np.asarray(node.get("translation", (0, 0, 0)))
        scale = np.asarray(node.get("scale", (1, 1, 1)))
        x, y, z, w = node.get("rotation", (0, 0, 0, 1))
        rotation = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation @ np.diag(scale)
        matrix[:3, 3] = translation
        return matrix

    @classmethod
    def _collect_meshes(cls, document, binary):
        positions = []
        uvs = []
        triangles = []

        def visit(node_index, parent_matrix):
            node = document["nodes"][node_index]
            matrix = parent_matrix @ cls._node_matrix(node)
            if "mesh" in node:
                mesh = document["meshes"][node["mesh"]]
                for primitive in mesh.get("primitives", []):
                    # Ignore line objects; the face model uses triangle meshes.
                    if primitive.get("mode", 4) != 4:
                        continue
                    attributes = primitive["attributes"]
                    if "POSITION" not in attributes or "TEXCOORD_0" not in attributes:
                        continue
                    points = cls._accessor(document, binary, attributes["POSITION"])
                    texcoords = cls._accessor(document, binary, attributes["TEXCOORD_0"])
                    points = (
                        np.c_[points, np.ones(len(points), dtype=np.float32)]
                        @ matrix.T
                    )[:, :3]
                    indices = cls._accessor(
                        document, binary, primitive["indices"]
                    ).reshape(-1)
                    faces = indices.reshape(-1, 3).astype(np.int64)
                    offset = sum(len(part) for part in positions)
                    positions.append(points.astype(np.float32))
                    uvs.append(texcoords.astype(np.float32))
                    triangles.append(faces + offset)

            for child in node.get("children", []):
                visit(child, matrix)

        root_nodes = document["scenes"][document.get("scene", 0)]["nodes"]
        for root in root_nodes:
            visit(root, np.eye(4, dtype=np.float64))

        if not positions or not triangles:
            raise ValueError("No se encontraron mallas triangulares con UV en el GLB.")
        return (
            np.concatenate(positions),
            np.concatenate(uvs),
            np.concatenate(triangles),
        )

    @staticmethod
    def _read_texture(document, binary):
        images = document.get("images", [])
        if not images:
            return None
        image = images[0]
        if "bufferView" not in image:
            return None
        view = document["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        encoded = binary[start:start + view["byteLength"]]
        decoded = cv2.imdecode(
            np.frombuffer(encoded, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if decoded is None:
            raise ValueError("No se pudo decodificar la textura del GLB.")
        return decoded

    def triangle_colors(self):
        """Obtiene colores de la textura para una aproximación rápida."""
        if self.texture is None:
            return np.tile((175, 155, 130), (len(self.triangles), 1))
        height, width = self.texture.shape[:2]
        uv = self.uvs[self.triangles].mean(axis=1)
        x = np.clip(np.rint(uv[:, 0] * (width - 1)).astype(int), 0, width - 1)
        y = np.clip(np.rint(uv[:, 1] * (height - 1)).astype(int), 0, height - 1)
        return self.texture[y, x]
