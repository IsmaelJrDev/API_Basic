import os
import io
import base64
from typing import Tuple

from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np

# Configuración
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def rgb_to_grayscale(img_np: np.ndarray) -> np.ndarray:
    """Convertir un arreglo numpy de imagen RGB a una imagen en escala de grises de 3 canales.

    Mantiene la forma de salida idéntica a la entrada (H, W, 3).
    """
    if img_np.ndim == 3:
        R, G, B = img_np[..., 0], img_np[..., 1], img_np[..., 2]
        grayscale_np = (0.2989 * R + 0.5870 * G + 0.1140 * B).astype(np.uint8)
        return np.stack([grayscale_np] * 3, axis=-1)
    return img_np


def sobel_filter(img_np: np.ndarray) -> np.ndarray:
    """Aplicar un filtro Sobel (magnitud del gradiente) a una imagen en escala de grises.

    La función espera un arreglo HxWx3, lo convierte a un único canal y calcula la
    magnitud del gradiente, devolviendo una imagen uint8 de 3 canales.
    """
    img_gray = rgb_to_grayscale(img_np)[..., 0]
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    rows, cols = img_gray.shape
    output = np.zeros_like(img_gray, dtype=np.float32)
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            patch = img_gray[y-1:y+2, x-1:x+2]
            Gx = np.sum(patch * Kx)
            Gy = np.sum(patch * Ky)
            output[y, x] = np.sqrt(Gx**2 + Gy**2)
    output = np.clip(output, 0, 255).astype(np.uint8)
    return np.stack([output] * 3, axis=-1)


def apply_full_transformation(
    img: Image.Image,
    rotation: float,
    scale: float,
    tx: float,
    ty: float,
    shearX: float,
    shearY: float,
    mirror: bool,
    flip: bool,
) -> Tuple[Image.Image, Tuple[float, float, float, float, float, float]]:
    """Aplicar transformaciones afines y devolver (imagen_transformada, matriz_2x3_pil).

    La matriz se devuelve en el formato de tupla de 6 elementos de PIL: (a, b, e, c, d, f).
    """
    # >>> OPERACIÓN MATRIZ: iniciar con la matriz homogénea identidad (3x3)
    M = np.eye(3)

    # >>> OPERACIÓN MATRIZ: matriz de escalado S y composición M = S @ M
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
    M = S @ M

    # >>> OPERACIÓN MATRIZ: aplicar reflexión horizontal/vertical (invertir signos diagonales)
    if mirror:
        M[0, 0] *= -1
    if flip:
        M[1, 1] *= -1

    # >>> OPERACIÓN MATRIZ: construir matriz de rotación R y componer M = R @ M
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)
    R = np.array([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]])
    M = R @ M

    # >>> OPERACIÓN MATRIZ: matriz de cizallamiento H y composición M = H @ M
    H = np.array([[1, shearX, 0], [shearY, 1, 0], [0, 0, 1]])
    M = H @ M

    width, height = img.size
    cx, cy = width / 2.0, height / 2.0

    # >>> OPERACIÓN MATRIZ: calcular desplazamiento desde el centro transformado y construir la tupla 2x3 para PIL
    center_transformed = M @ np.array([cx, cy, 1.0])
    e = tx + (cx - center_transformed[0])
    f = ty + (cy - center_transformed[1])

    # Matriz 2x3 en formato PIL: (a, b, e, c, d, f)
    transform_matrix_pil = (M[0, 0], M[0, 1], e, M[1, 0], M[1, 1], f)

    transformed_img = img.transform(
        img.size, Image.Transform.AFFINE, transform_matrix_pil, resample=Image.Resampling.BICUBIC
    )

    return transformed_img, transform_matrix_pil


# -------------------------
# Helpers para ejemplos
# -------------------------
def rotation_matrix(theta: float) -> np.ndarray:
    """Construir matriz homogénea 3x3 de rotación para ángulo en radianes."""
    return np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])


def scale_matrix(s: float) -> np.ndarray:
    """Construir matriz homogénea 3x3 de escala uniforme."""
    return np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]])


def shear_matrix(sx: float, sy: float) -> np.ndarray:
    """Construir matriz homogénea 3x3 de cizallamiento (shearX, shearY)."""
    return np.array([[1.0, sx, 0.0], [sy, 1.0, 0.0], [0.0, 0.0, 1.0]])


def reflection_matrix(mirror: bool, flip: bool) -> np.ndarray:
    """Construir matriz homogénea 3x3 para reflejos horizontales/verticales."""
    M = np.eye(3)
    if mirror:
        M[0, 0] = -1.0
    if flip:
        M[1, 1] = -1.0
    return M


def compose_matrices(*mats: np.ndarray) -> np.ndarray:
    """Componer matrices homogéneas en el orden dado: resultado = mats[0] @ mats[1] @ ...

    Nota: en el resto del código usamos M = A @ M para componer por la izquierda. Aquí
    aplicamos el orden directo para claridad.
    """
    if not mats:
        return np.eye(3)
    M = np.eye(3)
    for mat in mats:
        M = mat @ M
    return M


def linear_and_pil_from_homogeneous(M: np.ndarray) -> Tuple[list, tuple]:
    """Extraer la parte lineal 2x2 y formar la tupla PIL (a,b,e,c,d,f) con e=f=0."""
    a = float(M[0, 0])
    b = float(M[0, 1])
    c = float(M[1, 0])
    d = float(M[1, 1])
    linear = [[a, b], [c, d]]
    pil_tuple = (a, b, 0.0, c, d, 0.0)
    return linear, pil_tuple


@app.route('/api/matrix_example', methods=['POST'])
def matrix_example():
    """Endpoint de ejemplo que devuelve matrices para varios casos: rotación pura, rotación+escala,
    rotación+escala+shear y mirror antes de rotar. Recibe JSON con campos opcionales:
    { rotation(deg), scale, shearX, shearY, mirror, flip }
    """
    data = request.get_json() or {}
    rot_deg = float(data.get('rotation', 45.0))
    rot = np.deg2rad(rot_deg)
    s = float(data.get('scale', 1.0))
    sx = float(data.get('shearX', 0.0))
    sy = float(data.get('shearY', 0.0))
    mirror = bool(data.get('mirror', False))
    flip = bool(data.get('flip', False))

    # Rotación pura
    R = rotation_matrix(rot)
    rot_linear, rot_pil = linear_and_pil_from_homogeneous(R)

    # Rotación + escala (aplicamos S primero, luego R como en tu flujo)
    S = scale_matrix(s)
    RS = compose_matrices(R, S)
    rs_linear, rs_pil = linear_and_pil_from_homogeneous(RS)

    # Rotación + escala + shear (S luego R luego H)
    H = shear_matrix(sx, sy)
    RSH = compose_matrices(H, R, S)
    rsh_linear, rsh_pil = linear_and_pil_from_homogeneous(RSH)

    # Mirror antes de rotar
    M_ref = reflection_matrix(mirror, flip)
    MrefR = compose_matrices(R, M_ref)
    mref_linear, mref_pil = linear_and_pil_from_homogeneous(MrefR)

    return jsonify({
        'rotation_deg': rot_deg,
        'rotation_only': {'linear': rot_linear, 'pil': [round(v, 6) for v in rot_pil]},
        'rotation_scale': {'linear': rs_linear, 'pil': [round(v, 6) for v in rs_pil]},
        'rotation_scale_shear': {'linear': rsh_linear, 'pil': [round(v, 6) for v in rsh_pil]},
        'mirror_then_rotate': {'linear': mref_linear, 'pil': [round(v, 6) for v in mref_pil]},
    })


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/api/process', methods=['POST'])
def process_image_api():
    """Punto de entrada API que recibe parámetros de transformación y devuelve imagen procesada y la matriz."""
    img_path = os.path.join('static', 'Stark.jpg')
    if not os.path.exists(img_path):
        return jsonify({'error': 'Imagen "Stark.jpg" no encontrada en /static'}), 500

    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Error al cargar la imagen base: {str(e)}'}), 500

    try:
        data = request.get_json() or {}
        rotation = np.deg2rad(float(data.get('rotation', 0)))
        scale = float(data.get('scale', 1.0))
        tx = float(data.get('translateX', 0))
        ty = float(data.get('translateY', 0))
        shearX = float(data.get('shearX', 0))
        shearY = float(data.get('shearY', 0))
        mirror = bool(data.get('mirror', False))
        flip = bool(data.get('flip', False))
        toggle_sobel = bool(data.get('toggleSobel', False))
    except Exception as e:
        return jsonify({'error': f'Parámetros inválidos: {str(e)}'}), 400

    try:
        transformed_img, matrix_pil = apply_full_transformation(
            img, rotation, scale, tx, ty, shearX, shearY, mirror, flip
        )

        img_np = np.array(transformed_img)
        processed_img_np = rgb_to_grayscale(img_np)
        if toggle_sobel:
            processed_img_np = sobel_filter(processed_img_np)

        final_img = Image.fromarray(processed_img_np)

        buffered = io.BytesIO()
        final_img.save(buffered, format='JPEG')
        img_data_uri = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"

        matrix_display = [round(float(val), 4) for val in matrix_pil]

        return jsonify({
            'image_data': img_data_uri,
            'width': final_img.width,
            'height': final_img.height,
            'matrix': matrix_display,
        })

    except Exception as e:
        print(f'Error de procesamiento: {e}')
        return jsonify({'error': f'Error de procesamiento interno: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)