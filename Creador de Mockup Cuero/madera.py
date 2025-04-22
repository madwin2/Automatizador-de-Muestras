import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

def generate_woodburn_extended_two_halos(texture_path, logo_path, burned_texture_path, output_path,
                                       max_box_size=670,
                                       burn_strength=0.7,
                                       burn_blur=5,
                                       halo1_size=15,
                                       halo1_intensity=0.4,
                                       halo2_size=25,
                                       halo2_intensity=0.2,
                                       logo_margin=50):
    """
    Genera un mockup con efecto de quemado y dos halos.
    
    Parámetros:
    - texture_path: ruta a la imagen de textura base
    - logo_path: ruta al logo
    - burned_texture_path: ruta a la textura quemada
    - output_path: ruta donde guardar el resultado
    - max_box_size: tamaño máximo del logo en píxeles
    - burn_strength: intensidad del efecto de quemado (0-1)
    - burn_blur: radio del desenfoque del quemado
    - halo1_size: tamaño del primer halo
    - halo1_intensity: intensidad del primer halo (0-1)
    - halo2_size: tamaño del segundo halo
    - halo2_intensity: intensidad del segundo halo (0-1)
    - logo_margin: margen adicional alrededor del logo
    """
    
    # Cargar imágenes
    texture = cv2.imread(texture_path)
    burned = cv2.imread(burned_texture_path)
    logo = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
    
    if texture is None:
        raise FileNotFoundError(f" No se pudo cargar la textura: {texture_path}")
    if burned is None:
        raise FileNotFoundError(f" No se pudo cargar la textura quemada: {burned_texture_path}")
    if logo is None:
        raise FileNotFoundError(f" No se pudo cargar el logo: {logo_path}")
    
    # Procesar logo
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
    
    # Añadir margen
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
    
    # Generar halos
    halo1 = gaussian_filter(logo_float, halo1_size)
    halo2 = gaussian_filter(logo_float, halo2_size)
    
    # Ubicación centrada
    texture_h, texture_w = texture.shape[:2]
    x_offset = max(0, (texture_w - target_width) // 2)
    y_offset = max(0, (texture_h - target_height) // 2)
    
    # Región de trabajo
    region = texture[y_offset:y_offset+target_height, x_offset:x_offset+target_width].copy()
    burned_region = burned[y_offset:y_offset+target_height, x_offset:x_offset+target_width].copy()
    
    # Convertir a HLS para manipular luminosidad
    region_hls = cv2.cvtColor(region, cv2.COLOR_BGR2HLS).astype(np.float32)
    burned_hls = cv2.cvtColor(burned_region, cv2.COLOR_BGR2HLS).astype(np.float32)
    
    # Aplicar halos
    region_hls[..., 1] += halo1 * 255 * halo1_intensity
    region_hls[..., 1] += halo2 * 255 * halo2_intensity
    region_hls[..., 1] = np.clip(region_hls[..., 1], 0, 255)
    
    # Suavizar transición del quemado
    burn_mask = cv2.GaussianBlur(logo_float, (burn_blur*2+1, burn_blur*2+1), 0)
    burn_mask = burn_mask * burn_strength
    
    # Combinar texturas
    combined_hls = region_hls * (1 - burn_mask[..., None]) + burned_hls * burn_mask[..., None]
    combined = cv2.cvtColor(combined_hls.astype(np.uint8), cv2.COLOR_HLS2BGR)
    
    # Aplicar sombras adicionales
    shadow = cv2.GaussianBlur(logo_float, (21, 21), 0) * 0.4
    combined = cv2.subtract(combined, (shadow[..., None] * 255).astype(np.uint8))
    
    # Integrar resultado
    output = texture.copy()
    output[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = combined
    
    cv2.imwrite(output_path, output)
    print(f" Mockup generado sin bordes cortados ({target_width}x{target_height}px)")

if __name__ == "__main__":
    try:
        generate_woodburn_extended_two_halos(
            texture_path="texture_path.png",
            logo_path="tierra.png",
            burned_texture_path="burned_texture_path.png",
            output_path="MP.png"
        )
    except Exception as e:
        print(f" Error: {e}")