import cv2
import numpy as np

def simulate_realistic_leather_embossing(texture_path, logo_path, output_path,
                                         max_box_size=670,
                                         depth_strength=20,
                                         shadow_blur=11,
                                         highlight_blur=21,
                                         shadow_intensity=0.3,
                                         highlight_intensity=0.07,
                                         logo_margin=50):
    texture = cv2.imread(texture_path)
    logo = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)

    if texture is None:
        raise FileNotFoundError(f" No se pudo cargar la textura: {texture_path}")
    if logo is None:
        raise FileNotFoundError(f" No se pudo cargar el logo: {logo_path}")

    _, logo_bin = cv2.threshold(logo, 200, 255, cv2.THRESH_BINARY_INV)
    if np.count_nonzero(logo_bin) == 0:
        raise ValueError(" El logo está vacío luego del umbral. Revisá el archivo.")

    coords = cv2.findNonZero(logo_bin)
    if coords is None:
        raise ValueError(" No se encontraron elementos en el logo.")

    x, y, w, h = cv2.boundingRect(coords)
    if w == 0 or h == 0:
        raise ValueError(" El logo tiene dimensiones inválidas (0 de ancho o alto).")

    logo_crop = logo_bin[y:y+h, x:x+w]

    # Añadir margen para evitar bordes recortados al hacer blur
    logo_padded = cv2.copyMakeBorder(
        logo_crop,
        top=logo_margin,
        bottom=logo_margin,
        left=logo_margin,
        right=logo_margin,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )

    padded_h, padded_w = logo_padded.shape[:2]

    # Escalar al tamaño máximo
    scale_w = max_box_size / padded_w
    scale_h = max_box_size / padded_h
    scale = min(scale_w, scale_h)

    target_width = int(padded_w * scale)
    target_height = int(padded_h * scale)

    logo_resized = cv2.resize(logo_padded, (target_width, target_height), interpolation=cv2.INTER_AREA)
    logo_float = logo_resized.astype(np.float32) / 255.0

    # Gradientes y relieves
    grad_x = cv2.Sobel(logo_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(logo_float, cv2.CV_32F, 0, 1, ksize=3)
    light = cv2.GaussianBlur(grad_y - grad_x, (highlight_blur, highlight_blur), 0, borderType=cv2.BORDER_REFLECT)
    shadow = cv2.GaussianBlur(grad_x + grad_y, (shadow_blur, shadow_blur), 0, borderType=cv2.BORDER_REFLECT)
    light = np.clip(light * highlight_intensity, -1, 1)
    shadow = np.clip(shadow * shadow_intensity, -1, 1)

    # Ubicación centrada
    texture_h, texture_w = texture.shape[:2]
    x_offset = max(0, (texture_w - target_width) // 2)
    y_offset = max(0, (texture_h - target_height) // 2)

    region = texture[y_offset:y_offset+target_height, x_offset:x_offset+target_width]
    hls = cv2.cvtColor(region, cv2.COLOR_BGR2HLS).astype(np.float32)

    hls[..., 1] -= logo_float * 20
    hls[..., 1] *= (1 - logo_float * depth_strength / 255.0)
    hls[..., 1] += light * 255
    hls[..., 1] -= shadow * 255
    hls[..., 1] = np.clip(hls[..., 1], 0, 255)

    mask = cv2.GaussianBlur((logo_float > 0.1).astype(np.float32), (15, 15), 0, borderType=cv2.BORDER_REFLECT)
    region_soft = cv2.GaussianBlur(region, (13, 13), 5, borderType=cv2.BORDER_REFLECT)
    combined = region * (1 - mask[..., None]) + region_soft * mask[..., None]
    combined = combined.astype(np.uint8)

    emboss = cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2BGR)
    final = cv2.addWeighted(emboss, 0.6, combined, 0.4, 0)

    output = texture.copy()
    output[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = final
    cv2.imwrite(output_path, output)
    print(f" Mockup generado sin bordes cortados ({target_width}x{target_height}px)")

if __name__ == "__main__":
    try:
        simulate_realistic_leather_embossing(
            texture_path="cuero2.png",
            logo_path="tierra.png",
            output_path="MP.png"
        )
    except Exception as e:
        print(f" Error: {e}")