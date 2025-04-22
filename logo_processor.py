import cv2
import numpy as np
from PIL import Image
import os
from typing import Dict, Any, List
import json

class LogoProcessor:
    def __init__(self):
        self.tools = {
            "ajustar_contraste": self._ajustar_contraste,
            "binarizar": self._binarizar,
            "eliminar_fondo": self._eliminar_fondo,
            "engrosar_trazos": self._engrosar_trazos,
            "invertir_colores": self._invertir_colores,
            "eliminar_ruido": self._eliminar_ruido
        }

    def process_logo(self, input_path: str, steps: List[Dict[str, Any]], output_path: str = None) -> str:
        """
        Procesa un logo según una secuencia de pasos definida.
        
        Args:
            input_path: Ruta al archivo de imagen de entrada
            steps: Lista de diccionarios con los pasos a ejecutar
            output_path: Ruta donde guardar el resultado (opcional)
            
        Returns:
            Ruta del archivo procesado
        """
        # Verificar que el archivo existe
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

        # Si no se especifica output_path, crear uno basado en el input
        if output_path is None:
            base_name = os.path.splitext(input_path)[0]
            output_path = f"{base_name}_processed.png"

        # Cargar la imagen
        image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {input_path}")

        # Si la imagen no tiene canal alpha, añadirlo
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        # Procesar cada paso
        for step in steps:
            herramienta = step['herramienta']
            parametros = step['parametros']

            if herramienta not in self.tools:
                raise ValueError(f"Herramienta no reconocida: {herramienta}")

            # Ejecutar la herramienta
            image = self.tools[herramienta](image, **parametros)

        # Guardar el resultado
        final_image = self._ensure_black_logo_white_background(current_image)
        final_image = self._final_background_check(final_image)
        cv2.imwrite(output_path, final_image)
        return output_path, {
            'usar_original': False,
            'pasos': resultados_pasos
        }

    def _ajustar_contraste(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Ajusta el contraste de la imagen."""
        # Convertir a float32 para evitar overflow
        image_float = image.astype(np.float32) / 255.0
        
        # Ajustar contraste
        adjusted = np.clip(image_float * factor, 0, 1)
        
        # Volver a uint8
        return (adjusted * 255).astype(np.uint8)

    def _binarizar(self, image: np.ndarray, threshold: int) -> np.ndarray:
        """Convierte la imagen a blanco y negro usando un umbral."""
        # Asegurar que la imagen está en escala de grises
        if len(image.shape) > 2:
            if image.shape[2] == 4:  # Si tiene canal alpha (BGRA)
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:  # Si es BGR
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Aplicar umbral
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Convertir de nuevo a BGRA si la imagen original tenía 4 canales
        if len(image.shape) > 2 and image.shape[2] == 4:
            binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGRA)
        elif len(image.shape) > 2 and image.shape[2] == 3:
            binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        return binary

    def _eliminar_fondo(self, image: np.ndarray) -> np.ndarray:
        """Elimina el fondo de la imagen dejándolo transparente."""
        # Convertir a RGBA si es necesario
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        # Aplicar umbral adaptativo
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Crear máscara alpha
        alpha = binary.copy()
        
        # Aplicar la máscara al canal alpha de la imagen original
        image[:, :, 3] = alpha

        return image

    def _engrosar_trazos(self, image: np.ndarray, pixels: int) -> np.ndarray:
        """Engrosa las líneas de la imagen."""
        # Crear kernel para dilatación
        kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), np.uint8)
        
        # Si la imagen tiene canal alpha, procesarlo por separado
        if len(image.shape) > 2 and image.shape[2] == 4:
            # Separar canales
            bgr = image[:, :, :3]
            alpha = image[:, :, 3]
            
            # Dilatar
            bgr = cv2.dilate(bgr, kernel, iterations=1)
            alpha = cv2.dilate(alpha, kernel, iterations=1)
            
            # Combinar canales
            result = cv2.merge([bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], alpha])
        else:
            result = cv2.dilate(image, kernel, iterations=1)
        
        return result

    def _invertir_colores(self, image: np.ndarray) -> np.ndarray:
        """Invierte los colores de la imagen."""
        # Si la imagen tiene canal alpha, preservarlo
        if len(image.shape) > 2 and image.shape[2] == 4:
            # Separar canales
            bgr = image[:, :, :3]
            alpha = image[:, :, 3]
            
            # Invertir solo los canales BGR
            bgr = cv2.bitwise_not(bgr)
            
            # Combinar canales
            return cv2.merge([bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], alpha])
        else:
            return cv2.bitwise_not(image)

    def _eliminar_ruido(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Elimina pequeñas imperfecciones de la imagen."""
        # Crear kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Si la imagen tiene canal alpha, procesarlo por separado
        if len(image.shape) > 2 and image.shape[2] == 4:
            # Separar canales
            bgr = image[:, :, :3]
            alpha = image[:, :, 3]
            
            # Aplicar operaciones morfológicas
            bgr = cv2.morphologyEx(bgr, cv2.MORPH_OPEN, kernel)
            alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
            
            # Combinar canales
            return cv2.merge([bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], alpha])
        else:
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

if __name__ == "__main__":
    # Ejemplo de uso
    processor = LogoProcessor()
    
    # Cargar los pasos desde un archivo JSON o definirlos programáticamente
    steps = [
        {
            "herramienta": "eliminar_fondo",
            "parametros": {}
        },
        {
            "herramienta": "binarizar",
            "parametros": {"threshold": 128}
        },
        {
            "herramienta": "eliminar_ruido",
            "parametros": {"kernel_size": 3}
        },
        {
            "herramienta": "engrosar_trazos",
            "parametros": {"pixels": 2}
        }
    ]
    
    try:
        # Procesar el logo
        output_path = processor.process_logo("prueba.jpg", steps)
        print(f"Logo procesado guardado en: {output_path}")
    except Exception as e:
        print(f"Error procesando el logo: {str(e)}")