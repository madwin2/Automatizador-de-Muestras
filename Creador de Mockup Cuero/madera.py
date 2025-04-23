import cv2
import numpy as np

def generate_woodburn_extended_two_halos(texture_path, logo_path, burned_texture_path, output_path,
                                         desired_logo_size=670,      # Tamaño para el área central del logo
                                         extra_margin=150,           # Margen adicional para que los halos no se recorten
                                         blur_radius=3,
                                         burn_intensity=1.1,
                                         relief_intensity=0.08,
                                         red_tint_intensity=25,
                                         # Parámetros del halo grande:
                                         halo_darken_factor=0.1,
                                         halo_color=np.array([80, 45, 15], dtype=np.uint8),
                                         halo_exponent=8,
                                         # Parámetros para el halo chico:
                                         second_halo_scale=0.1,         # Menor valor: el halo chico se mantendrá más pegado
                                         second_halo_intensity=0.6,     # Factor de intensidad del halo chico
                                         second_halo_color=np.array([80, 45, 15], dtype=np.uint8),  # Modifica este valor al color de quema deseado
                                         second_halo_darken_factor=0.03,
                                         noise_strength=0.4):
    """
    Genera un mockup con efecto de quemado y dos halos:
      - Un halo grande (más difuso).
      - Un halo chico (más concentrado e intenso).
    
    El logo se escala sin deformarse para que ocupe un área central de 'desired_logo_size'
    (ej. 670x670) y luego se inserta en un lienzo mayor para que ambos halos se extiendan sin recortarse.
    
    Para modificar el color del halo chico, modifica la variable second_halo_color.
    """
    ### 1. Cargar imágenes
    texture = cv2.imread(texture_path)
    burned_texture = cv2.imread(burned_texture_path)
    # 1. Cargar logo y escalarlo ANTES de binarizar (para que mantenga suavidad)
    logo_gray_orig = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
    if logo_gray_orig is None:
        raise FileNotFoundError("Logo no encontrado.")

    # Aumentar tamaño del logo original si es chico (acá por ejemplo a 1000 px de ancho)
    max_dim = max(logo_gray_orig.shape)
    if max_dim < 1000:
        upscale_factor = 1000 / max_dim
        new_size = (int(logo_gray_orig.shape[1] * upscale_factor), int(logo_gray_orig.shape[0] * upscale_factor))
        logo_gray = cv2.resize(logo_gray_orig, new_size, interpolation=cv2.INTER_CUBIC)
    else:
        logo_gray = logo_gray_orig.copy()

    # 2. Binarizar suavemente después del upscale
    logo_blur = cv2.GaussianBlur(logo_gray, (3,3), 0)
    _, logo_bin = cv2.threshold(logo_blur, 180, 255, cv2.THRESH_BINARY_INV)

    # 3. Recortar y reescalar a destino
    coords = cv2.findNonZero(logo_bin)
    x, y, w, h = cv2.boundingRect(coords)
    logo_crop = logo_bin[y:y+h, x:x+w]

    scale = min(desired_logo_size / w, desired_logo_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    logo_resized = cv2.resize(logo_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # (Opcional) suavizado con cierre morfológico
    kernel = np.ones((3, 3), np.uint8)
    logo_resized = cv2.morphologyEx(logo_resized, cv2.MORPH_CLOSE, kernel)

    # 4. Posicionar en el lienzo final
    small_canvas = np.zeros((desired_logo_size, desired_logo_size), dtype=np.uint8)
    offset_x_small = (desired_logo_size - new_w) // 2
    offset_y_small = (desired_logo_size - new_h) // 2
    small_canvas[offset_y_small:offset_y_small+new_h, offset_x_small:offset_x_small+new_w] = logo_resized
    small_mask = small_canvas.astype(np.float32) / 255.0


    ### 3. Aplicar efecto de quemado en el canvas pequeño (670×670)
    small_blur = cv2.GaussianBlur(small_mask, (blur_radius, blur_radius), 0)
    noise = np.random.normal(loc=1.0, scale=noise_strength, size=small_mask.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (31, 31), 0)
    burn_variation = np.clip(small_blur * noise, 0, 1)
    
    # Bump map mediante derivadas
    grad_x = cv2.Sobel(small_mask, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(small_mask, cv2.CV_32F, 0, 1, ksize=3)
    bump_map = cv2.GaussianBlur(grad_x + grad_y, (7, 7), 0)
    bump_map = np.clip(bump_map * relief_intensity * 255, -20, 20).astype(np.int16)
    
    burned_small = cv2.resize(burned_texture, (desired_logo_size, desired_logo_size), interpolation=cv2.INTER_AREA)
    burn_variation_3ch = np.stack([burn_variation]*3, axis=-1)
    base_dummy = np.zeros((desired_logo_size, desired_logo_size, 3), dtype=np.uint8)
    burned_mix = (base_dummy * (1 - burn_variation_3ch * burn_intensity) +
                    burned_small * (burn_variation_3ch * burn_intensity)).astype(np.uint8)
    darkening_mask = (1 - burn_variation[..., None] * 0.2).astype(np.float32)
    burned_mix = (burned_mix.astype(np.float32) * darkening_mask).clip(0,255).astype(np.uint8)
    
    # Aplicar tinte rojizo:
    red_tint = burned_mix.copy()
    red_tint[:, :, 2] = np.clip(red_tint[:, :, 2] + red_tint_intensity * burn_variation, 0, 255)
    hls = cv2.cvtColor(red_tint, cv2.COLOR_BGR2HLS).astype(np.int16)
    hls[:, :, 1] = np.clip(hls[:, :, 1] - bump_map, 0, 255)
    small_final_stamp = cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2BGR)
    
    ### 4. Crear un lienzo mayor para que los halos no se recorten
    big_canvas_size = desired_logo_size + 2 * extra_margin  # e.g., 670 + 300 = 970
    # Canvas para la máscara y para la imagen quemada:
    big_mask = np.zeros((big_canvas_size, big_canvas_size), dtype=np.float32)
    big_final_stamp = np.zeros((big_canvas_size, big_canvas_size, 3), dtype=np.uint8)
    # Colocar el canvas pequeño centrado en el lienzo grande:
    offset_big = extra_margin
    big_mask[offset_big:offset_big+desired_logo_size, offset_big:offset_big+desired_logo_size] = small_mask
    big_final_stamp[offset_big:offset_big+desired_logo_size, offset_big:offset_big+desired_logo_size] = small_final_stamp
    
    ### 5. Generar las máscaras de halo a partir de la máscara en el lienzo grande
    # Se obtiene la transformada de distancia en la región fuera del logo:
    big_mask_bin = (big_mask > 0.5).astype(np.uint8)
    inv_big_mask_bin = 1 - big_mask_bin
    dist = cv2.distanceTransform((inv_big_mask_bin * 255).astype(np.uint8), cv2.DIST_L2, 3)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    
    # --- Halo grande ---
    # Ajusta max_allowed para controlar la extensión
    max_allowed = 0.02  
    dist_clamped = np.clip(dist_norm, 0, max_allowed) / max_allowed
    big_halo_mask = np.exp(-halo_exponent * dist_clamped)
    
    # --- Halo chico ---
    # Se genera a partir de la misma distancia, pero escalada para estar más concentrado
    second_halo_mask = np.exp(-halo_exponent * np.clip(dist_norm * second_halo_scale, 0, max_allowed) / max_allowed)
    second_halo_mask = np.clip(second_halo_mask * second_halo_intensity, 0, 1)
    
    # Crear las capas de color para cada halo:
    big_halo_layer = (big_halo_mask[..., None] * halo_color.astype(np.float32)).clip(0,255).astype(np.uint8)
    second_halo_layer = (second_halo_mask[..., None] * second_halo_color.astype(np.float32)).clip(0,255).astype(np.uint8)
    
    # Combinar ambos halos (la forma de combinación se puede ajustar; aquí se usa addWeighted)
    combined_halo = (
    big_halo_layer.astype(np.float32) * 0.1 +
    second_halo_layer.astype(np.float32) * 0.3  # Usamos 1.0 para que el halo chico no se diluya tanto
    ).clip(0, 255).astype(np.uint8)

    
    ### 6. Construir la máscara alfa final
    # La alfa es la unión del logo quemado (big_mask) y de ambos halos:
    alpha_final = np.clip(big_mask + big_halo_mask + second_halo_mask, 0, 1)
    alpha_final[alpha_final < 0.01] = 0  # Forzamos valores muy bajos a 0
    alpha_final_3ch = np.dstack([alpha_final, alpha_final, alpha_final])
    
    # Combinar la imagen quemada (big_final_stamp) y la capa combinada de halos:
    big_mask_3ch = np.dstack([big_mask, big_mask, big_mask])
    composite_big = (big_final_stamp.astype(np.float32) * big_mask_3ch +
                     combined_halo.astype(np.float32) * (1 - big_mask_3ch))
    composite_big = composite_big.clip(0,255).astype(np.uint8)
    composite_big = (composite_big.astype(np.float32) * alpha_final_3ch).clip(0,255).astype(np.uint8)
    
    ### 7. Componer el sello (composite_big) sobre la textura original
    out = texture.copy()
    th, tw = out.shape[:2]
    center_x = tw // 2
    center_y = th // 2
    x_offset = center_x - (big_canvas_size // 2)
    y_offset = center_y - (big_canvas_size // 2)
    
    # Ajustar límites para que no se salga de la textura
    x1 = max(0, x_offset)
    y1 = max(0, y_offset)
    x2 = min(tw, x_offset + big_canvas_size)
    y2 = min(th, y_offset + big_canvas_size)
    
    # Recortar la porción del sello y de la máscara alfa
    crop_x1 = x1 - x_offset
    crop_y1 = y1 - y_offset
    crop_x2 = crop_x1 + (x2 - x1)
    crop_y2 = crop_y1 + (y2 - y1)
    composite_crop = composite_big[crop_y1:crop_y2, crop_x1:crop_x2]
    alpha_crop = alpha_final[crop_y1:crop_y2, crop_x1:crop_x2]
    alpha_crop_3ch = np.dstack([alpha_crop, alpha_crop, alpha_crop])
    
    composite_crop_f = composite_crop.astype(np.float32) / 255.0
    texture_region_f = out[y1:y2, x1:x2].astype(np.float32) / 255.0
    blended_region = (texture_region_f * (1 - alpha_crop_3ch) + composite_crop_f * alpha_crop_3ch)
    blended_region = (blended_region * 255).clip(0,255).astype(np.uint8)
    
    out[y1:y2, x1:x2] = blended_region
    cv2.imwrite(output_path, out)
    print(" Mockup generado con halo grande y halo chico.")

if __name__ == "__main__":
    generate_woodburn_extended_two_halos(
        texture_path="texture_path.png",         # Ruta de la textura
        logo_path="Overol.jpg",                 # Ruta del logo
        burned_texture_path="burned_texture_path.png",  # Ruta de la textura quemada
        output_path="mockup_two_halos.png"        # Ruta de salida
    )
