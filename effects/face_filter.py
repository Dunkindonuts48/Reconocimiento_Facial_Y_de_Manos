"""Renderizado acelerado de una cabeza GLB sobre la cámara."""

import cv2
import moderngl
import numpy as np

from effects.glb_face_model import GLBFaceModel


FACE_POSE_POINTS = {
    10: (0.0, -90.0, -8.0),
    152: (0.0, 100.0, 0.0),
    234: (-82.0, 0.0, -8.0),
    454: (82.0, 0.0, -8.0),
    33: (-34.0, -25.0, 24.0),
    263: (34.0, -25.0, 24.0),
    1: (0.0, 20.0, 48.0),
}


class FaceModelFilter:
    """Dibuja el GLB en GPU con profundidad real y conserva los efectos de mano."""

    def __init__(self, model_path):
        self.model = GLBFaceModel(model_path)
        # glcontext selecciona WGL por defecto en Windows. Su argumento
        # backend= admite nombres como "egl", pero no "wgl".
        try:
            self.ctx = moderngl.create_standalone_context(require=330)
        except Exception as error:
            raise RuntimeError(
                "No se pudo crear un contexto OpenGL. Actualiza el controlador "
                "gráfico y comprueba que admite OpenGL 3.3."
            ) from error
        self.program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                in vec2 in_uv;
                uniform mat4 u_mvp;
                out vec2 v_uv;
                void main() {
                    gl_Position = u_mvp * vec4(in_position, 1.0);
                    v_uv = in_uv;
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D u_texture;
                in vec2 v_uv;
                out vec4 frag_color;
                void main() {
                    vec3 color = texture(u_texture, v_uv).rgb;
                    // La textura del escaneo es un atlas JPEG sobre negro;
                    // sus texeles vacíos deben dejar pasar la cámara.
                    if (max(max(color.r, color.g), color.b) < 0.035) {
                        discard;
                    }
                    frag_color = vec4(color, 1.0);
                }
            """,
        )
        interleaved = np.column_stack((self.model.vertices, self.model.uvs)).astype("f4")
        self.vertex_buffer = self.ctx.buffer(interleaved.tobytes())
        self.index_buffer = self.ctx.buffer(self.model.triangles.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vertex_buffer, "3f 2f", "in_position", "in_uv")],
            index_buffer=self.index_buffer,
            index_element_size=4,
        )
        texture = self.model.texture
        if texture is None:
            texture = np.full((2, 2, 3), (130, 155, 175), dtype=np.uint8)
        rgb = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
        self.texture = self.ctx.texture(
            (rgb.shape[1], rgb.shape[0]), 3, np.ascontiguousarray(rgb).tobytes()
        )
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.program["u_texture"].value = 0
        self._target_size = None
        self._framebuffer = None
        self._color = None
        self._depth = None
        self.status = "esperando cara"

    @staticmethod
    def _camera_matrix(frame):
        height, width = frame.shape[:2]
        focal_length = float(width) * 1.15
        return np.asarray(
            [[focal_length, 0.0, width / 2.0],
             [0.0, focal_length, height / 2.0],
             [0.0, 0.0, 1.0]], dtype=np.float64
        )

    def _face_pose(self, frame, landmarks):
        height, width = frame.shape[:2]
        image_points = np.asarray(
            [(landmarks[index].x * width, landmarks[index].y * height)
             for index in FACE_POSE_POINTS], dtype=np.float64
        )
        object_points = np.asarray(list(FACE_POSE_POINTS.values()), dtype=np.float64)
        camera = self._camera_matrix(frame)
        distortion = np.zeros((4, 1), dtype=np.float64)
        flags = [cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_EPNP]
        if hasattr(cv2, "SOLVEPNP_SQPNP"):
            flags.append(cv2.SOLVEPNP_SQPNP)

        candidates = []
        for flag in flags:
            try:
                success, rotation, translation = cv2.solvePnP(
                    object_points, image_points, camera, distortion, flags=flag
                )
            except cv2.error:
                continue
            if not success:
                continue

            rotation_matrix, _ = cv2.Rodrigues(rotation)
            camera_points = (rotation_matrix @ object_points.T + translation).T
            front_count = int(np.count_nonzero(camera_points[:, 2] > 1.0))
            projected, _ = cv2.projectPoints(
                object_points, rotation, translation, camera, distortion
            )
            error = float(np.mean(np.linalg.norm(
                projected.reshape(-1, 2) - image_points, axis=1
            )))
            # Entre soluciones con error similar, evita la pose invertida o
            # lateral: el modelo parte mirando hacia +Z, como la cámara.
            facing_angle = float(np.arccos(np.clip(rotation_matrix[2, 2], -1.0, 1.0)))
            score = error + 8.0 * facing_angle
            candidates.append((front_count, score, rotation, translation))

        if not candidates:
            return None
        # Primero se exige la solución con más puntos delante del sensor; si
        # varias cumplen, se elige la que mejor reproduce los landmarks.
        front_count, _, rotation, translation = min(
            candidates, key=lambda item: (-item[0], item[1])
        )
        if front_count < len(object_points) // 2 + 1:
            return None
        return rotation, translation, camera, distortion

    def _ensure_target(self, size):
        if self._target_size == size:
            return
        if self._framebuffer is not None:
            self._framebuffer.release()
            self._color.release()
            self._depth.release()
        width, height = size
        self._color = self.ctx.texture(size, 4, dtype="f1")
        self._depth = self.ctx.depth_renderbuffer(size)
        self._framebuffer = self.ctx.framebuffer(
            color_attachments=[self._color], depth_attachment=self._depth
        )
        self._target_size = size

    def _render_textured_mesh(self, frame, pose, target_box, adjustments):
        rotation, translation, camera, _ = pose
        height, width = frame.shape[:2]
        near, far = 1.0, 10000.0
        fx, fy = camera[0, 0], camera[1, 1]
        cx, cy = camera[0, 2], camera[1, 2]
        projection = np.asarray(
            [[2 * fx / width, 0, 2 * cx / width - 1, 0],
             [0, -2 * fy / height, 1 - 2 * cy / height, 0],
             [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)],
             [0, 0, 1, 0]], dtype=np.float32
        )
        base_rotation, _ = cv2.Rodrigues(rotation)
        pitch = np.arctan2(base_rotation[2, 1], base_rotation[2, 2])
        yaw = np.arcsin(np.clip(-base_rotation[2, 0], -1.0, 1.0))
        roll = np.arctan2(base_rotation[1, 0], base_rotation[0, 0])
        if adjustments["invert_yaw"]:
            yaw = -yaw
        pitch += np.deg2rad(adjustments["pitch"])
        yaw += np.deg2rad(adjustments["yaw"])
        roll += np.deg2rad(adjustments["roll"])
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cr, sr = np.cos(roll), np.sin(roll)
        rotate_x = np.asarray([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        rotate_y = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rotate_z = np.asarray([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        rotation_matrix = (rotate_z @ rotate_y @ rotate_x).astype(np.float32)
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = rotation_matrix
        view[:3, 3] = translation.reshape(3)

        # La escala/origen del GLB no coincide necesariamente con los puntos
        # faciales genéricos usados por solvePnP. Reencuadramos la proyección
        # de la cabeza dentro del rectángulo facial observado.
        stride = max(1, len(self.model.vertices) // 5000)
        sample = self.model.vertices[::stride].astype(np.float64)
        camera_points = (rotation_matrix @ sample.T + translation).T
        in_front = camera_points[:, 2] > near
        if not np.any(in_front):
            self._render_diagnostic = f"sin geometría delante; z={camera_points[:, 2].min():.0f}"
            return 0
        camera_points = camera_points[in_front]
        depth_min = float(camera_points[:, 2].min())
        depth_max = float(camera_points[:, 2].max())
        screen_x = fx * camera_points[:, 0] / camera_points[:, 2] + cx
        screen_y = fy * camera_points[:, 1] / camera_points[:, 2] + cy
        model_left, model_right = float(screen_x.min()), float(screen_x.max())
        model_top, model_bottom = float(screen_y.min()), float(screen_y.max())
        model_center_x = (model_left + model_right) * 0.5
        model_center_y = (model_top + model_bottom) * 0.5
        target_left, target_top, target_right, target_bottom = target_box
        target_center_x = (target_left + target_right) * 0.5 + adjustments["x"]
        target_center_y = (target_top + target_bottom) * 0.5 + adjustments["y"]
        scale = adjustments["scale"] / 100.0
        target_width = (target_right - target_left) * scale
        target_height = (target_bottom - target_top) * scale
        target_left = target_center_x - target_width * 0.5
        target_right = target_center_x + target_width * 0.5
        target_top = target_center_y - target_height * 0.5
        target_bottom = target_center_y + target_height * 0.5
        scale_x = max(target_right - target_left, 1.0) / max(model_right - model_left, 1.0)
        scale_y = max(target_bottom - target_top, 1.0) / max(model_bottom - model_top, 1.0)
        target_center_x = (target_left + target_right) * 0.5
        target_center_y = (target_top + target_bottom) * 0.5
        offset_x = target_center_x - scale_x * model_center_x
        offset_y = target_center_y - scale_y * model_center_y
        projection[0, :] = scale_x * projection[0, :] + (
            scale_x - 1.0 + 2.0 * offset_x / width
        ) * projection[3, :]
        projection[1, :] = scale_y * projection[1, :] + (
            1.0 - scale_y - 2.0 * offset_y / height
        ) * projection[3, :]
        mvp = projection @ view

        self._ensure_target((width, height))
        self._framebuffer.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self._framebuffer.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        self.program["u_mvp"].write(mvp.T.tobytes())
        self.texture.use(location=0)
        self.vao.render(mode=moderngl.TRIANGLES)

        rgba = np.frombuffer(self._framebuffer.read(components=4, alignment=1), dtype=np.uint8)
        rgba = rgba.reshape(height, width, 4)[::-1]
        visible = rgba[:, :, 3] > 0
        frame[visible] = rgba[:, :, 2::-1][visible]
        self._render_diagnostic = (
            f"z={depth_min:.0f}..{depth_max:.0f} "
            f"xy={model_left:.0f},{model_top:.0f}:{model_right:.0f},{model_bottom:.0f} "
            f"idx={self.vao.vertices} gl={self.ctx.error}"
        )
        return int(np.count_nonzero(visible))

    @staticmethod
    def _draw_repulsor(frame, landmarks):
        height, width = frame.shape[:2]
        palm_indices = (0, 5, 9, 13, 17)
        palm = np.asarray(
            [(landmarks[i].x * width, landmarks[i].y * height)
             for i in palm_indices], dtype=np.float32
        )
        center = tuple(int(value) for value in np.rint(palm.mean(axis=0)))
        radius = max(8, int(np.linalg.norm(palm[1] - palm[4]) * 0.17))
        glow = np.zeros_like(frame)
        cv2.circle(glow, center, radius, (235, 164, 57), -1, cv2.LINE_AA)
        blur_size = max(9, (radius // 2) * 2 + 1)
        glow = cv2.GaussianBlur(glow, (blur_size, blur_size), radius * 0.3)
        cv2.addWeighted(frame, 1.0, glow, 0.8, 0.0, dst=frame)
        cv2.circle(frame, center, radius, (248, 177, 62), 2, cv2.LINE_AA)
        cv2.circle(frame, center, int(radius * 0.65), (255, 236, 172), 2, cv2.LINE_AA)
        cv2.circle(frame, center, max(2, int(radius * 0.28)), (255, 255, 250), -1, cv2.LINE_AA)

    def draw(self, frame, face_results, hand_results, adjustments=None):
        if adjustments is None:
            adjustments = {
                "x": 0, "y": 0, "scale": 100,
                "yaw": 0, "pitch": 0, "roll": 0, "invert_yaw": False,
            }
        if face_results.face_landmarks:
            landmarks = face_results.face_landmarks[0]
            if len(landmarks) > max(FACE_POSE_POINTS):
                pose = self._face_pose(frame, landmarks)
                if pose is not None:
                    height, width = frame.shape[:2]
                    face_top = landmarks[10].y * height
                    face_bottom = landmarks[152].y * height
                    face_left = landmarks[234].x * width
                    face_right = landmarks[454].x * width
                    face_width = max(face_right - face_left, 1.0)
                    face_height = max(face_bottom - face_top, 1.0)
                    target_box = (
                        face_left - face_width * 0.08,
                        face_top - face_height * 0.12,
                        face_right + face_width * 0.08,
                        face_bottom + face_height * 0.04,
                    )
                    pixel_count = self._render_textured_mesh(
                        frame, pose, target_box, adjustments
                    )
                    self.status = (
                        f"modelo visible ({pixel_count} px)"
                        if pixel_count else f"0 px | {self._render_diagnostic}"
                    )
                else:
                    self.status = "no se pudo calcular la pose facial"
            else:
                self.status = "landmarks faciales incompletos"
        else:
            self.status = "MediaPipe no detecta una cara"
        if hand_results.hand_landmarks:
            for hand in hand_results.hand_landmarks:
                self._draw_repulsor(frame, hand)
        return frame

    def close(self):
        for resource in (self._framebuffer, self._color, self._depth,
                         self.vao, self.index_buffer, self.vertex_buffer,
                         self.texture, self.program, self.ctx):
            if resource is not None:
                resource.release()
