import cv2
import numpy as np
import requests
import os

def analyze_and_process_logo(input_path, api_key):
    # Cargar la imagen
    image = cv2.imread(input_path)

    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calcular la desviación estándar y complejidad de textura
    std_dev = float(np.std(gray))
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_complexity = float(np.std(laplacian))

    # Verificar color del borde
    borders = np.concatenate([
        image[0, :, :], image[-1, :, :], image[:, 0, :], image[:, -1, :]
    ])
    mean_border_color = np.mean(borders, axis=0)
    border_is_white = np.all(mean_border_color > 220)

    # Verificar si hay elementos tocando los bordes
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    touching_borders = sum(
        1 for c in contours 
        if cv2.boundingRect(c)[0] == 0 or cv2.boundingRect(c)[1] == 0 or 
           cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] >= image.shape[1] or
           cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] >= image.shape[0]
    )

    # Nuevo criterio más robusto
    is_complex_background = int(
        (std_dev > 35 and texture_complexity > 8) or not border_is_white or touching_borders > 0
    )

    preanalysis_result = {
        'is_complex_background': is_complex_background,
        'std_dev': round(std_dev, 2),
        'texture_complexity': round(texture_complexity, 2),
        'border_is_white': int(border_is_white),
        'touching_borders': touching_borders
    }

    # Si el fondo es complejo, usar PhotoRoom
    if is_complex_background == 1:
        try:
            print(f"Procesando imagen con PhotoRoom: {input_path}")
            with open(input_path, 'rb') as image_file:
                files = {'image_file': ('image.jpg', image_file, 'image/jpeg')}
                headers = {'x-api-key': '2ee6de145c9f881084268fbb7319ef4096f82e7a'}
                response = requests.post(
                    'https://sdk.photoroom.com/v1/segment',
                    files=files,
                    headers=headers
                )

            if response.status_code != 200:
                print(f"Error en PhotoRoom API: {response.status_code}")
                print(f"Respuesta: {response.text}")
                return input_path, preanalysis_result

            result_image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)
            if result_image is None or result_image.shape[2] != 4:
                print("Error: No se pudo decodificar la imagen o falta canal alpha")
                return input_path, preanalysis_result

            height, width = result_image.shape[:2]
            alpha_mask = result_image[:, :, 3] / 255.0
            alpha_mask = np.stack([alpha_mask] * 3, axis=-1)
            foreground = result_image[:, :, :3]

            # Determinar si el logo es blanco (muy claro)
            gray_foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray_foreground)
            print(f"Intensidad media del logo: {mean_intensity}")

            # Siempre aplicar fondo blanco, excepto si el logo es muy claro (blanco)
            if mean_intensity > 235:
                bg_color = [0, 0, 0]
                print("⚠️ Logo blanco detectado. Fondo negro aplicado para visibilidad.")
            else:
                bg_color = [255, 255, 255]
                print("Fondo blanco aplicado.")


            custom_background = np.ones((height, width, 3), dtype=np.uint8)
            custom_background[:] = bg_color

            result_image = (foreground * alpha_mask + custom_background * (1 - alpha_mask)).astype(np.uint8)

            # Guardar resultado
            processed_path = os.path.join('temp', os.path.basename(input_path))
            cv2.imwrite(processed_path, result_image)
            print(f"Imagen procesada guardada en: {processed_path}")

            return processed_path, preanalysis_result

        except Exception as e:
            print(f"Error procesando con PhotoRoom: {str(e)}")
            return input_path, preanalysis_result

    # Si el fondo no es complejo, devolver imagen original
    return input_path, preanalysis_result
