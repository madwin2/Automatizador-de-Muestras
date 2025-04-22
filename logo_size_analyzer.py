import cv2
import numpy as np
from PIL import Image, ImageFont
import pytesseract
from typing import Tuple, Dict, List, Any
import os

class LogoSizeAnalyzer:
    def __init__(self):
        self.MM_TO_PIXEL_RATIO = 11.811  # 1mm = 11.811 pixels (300 DPI)
        
    def _convert_mm_to_pixels(self, mm: float) -> int:
        return int(mm * self.MM_TO_PIXEL_RATIO)
        
    def _pixels_to_mm(self, pixels: float) -> float:
        return pixels / self.MM_TO_PIXEL_RATIO
        
    def _detect_logo_boundaries(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """Detecta los límites del logo usando múltiples técnicas."""
        # Verificar que la imagen sea válida
        if not isinstance(image, np.ndarray):
            raise ValueError("La imagen debe ser un array de numpy")
        
        # Convertir a escala de grises considerando canal alfa
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # BGRA
                # Usar canal alfa si está disponible
                alpha = image[:, :, 3]
                # Combinar canal alfa con luminancia
                bgr = image[:, :, :3]
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                # Usar una combinación ponderada de alfa y luminancia
                gray = cv2.addWeighted(gray, 0.7, alpha, 0.3, 0)
            else:  # BGR
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. Método de umbral adaptativo
        thresh_adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # 2. Método Otsu
        _, thresh_otsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # 3. Método de detección de bordes
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3,3), np.uint8)  # Kernel más pequeño
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)

        # Combinar los tres métodos
        combined = cv2.bitwise_or(thresh_adaptive, thresh_otsu)
        combined = cv2.bitwise_or(combined, edges_dilated)

        # Limpiar ruido
        kernel_clean = np.ones((2,2), np.uint8)  # Kernel más pequeño para limpieza
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_clean)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_clean)

        # Encontrar contornos
        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            # Si no se encuentran contornos, intentar con umbral simple
            _, thresh_simple = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(
                thresh_simple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                raise ValueError("No se detectaron elementos del logo")

        # Filtrar contornos por área
        min_area = image.shape[0] * image.shape[1] * 0.0005  # Reducido a 0.05% del área total
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        if not valid_contours:
            valid_contours = contours  # Usar todos los contornos si ninguno pasa el filtro

        # Encontrar el rectángulo delimitador que incluye todos los contornos válidos
        x_min = y_min = float('inf')
        x_max = y_max = float('-inf')
        
        for cnt in valid_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        # Convertir a enteros y agregar un margen más pequeño
        margin_x = int((x_max - x_min) * 0.02)  # 2% del ancho
        margin_y = int((y_max - y_min) * 0.02)  # 2% del alto

        x = max(0, int(x_min) - margin_x)
        y = max(0, int(y_min) - margin_y)
        w = min(image.shape[1] - x, int(x_max - x_min) + 2 * margin_x)
        h = min(image.shape[0] - y, int(y_max - y_min) + 2 * margin_y)

        return (x, y, w, h)
    
    def _detect_text_heights(self, image: np.ndarray) -> List[float]:
        """Detecta las alturas del texto en la imagen."""
        try:
            # Asegurarse de que la imagen sea un array numpy válido
            if not isinstance(image, np.ndarray):
                raise ValueError("La imagen debe ser un array de numpy")
            
            # Hacer una copia de la imagen
            img_cv = image.copy()
            
            # Convertir a escala de grises si es necesario
            if len(img_cv.shape) == 3:
                if img_cv.shape[2] == 4:  # BGRA
                    # Separar canales BGR y alfa
                    bgr = img_cv[:, :, :3]
                    alpha = img_cv[:, :, 3]
                    # Convertir BGR a escala de grises
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    # Combinar con canal alfa
                    gray = cv2.addWeighted(gray, 0.7, alpha, 0.3, 0)
                else:  # BGR
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_cv.copy()
            
            # Asegurarse de que sea uint8
            if gray.dtype != np.uint8:
                gray = (gray * 255).astype(np.uint8)
            
            # Mejorar el contraste
            if np.mean(gray) > 127:  # Si la imagen es mayormente clara
                gray = cv2.bitwise_not(gray)

            # Aplicar umbral adaptativo con parámetros más estrictos
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                15,  # Bloque más grande
                5    # Constante C más alta
            )

            # Limpiar ruido y elementos pequeños
            kernel = np.ones((2,2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Encontrar contornos para filtrado inicial
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar contornos por área y forma
            min_area = gray.size * 0.0005  # Área mínima (0.05% de la imagen)
            max_area = gray.size * 0.2     # Área máxima (20% de la imagen)
            text_mask = np.zeros_like(binary)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if min_area < area < max_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = w / float(h)
                    # Filtrar por proporción de aspecto más estricta para texto
                    if 0.1 < aspect_ratio < 15:
                        # Calcular la densidad del contorno
                        roi = binary[y:y+h, x:x+w]
                        density = np.count_nonzero(roi) / (w * h)
                        # El texto suele tener una densidad media
                        if 0.2 < density < 0.8:
                            cv2.drawContours(text_mask, [cnt], -1, 255, -1)

            # Convertir a formato PIL para Tesseract
            img_pil = Image.fromarray(text_mask)
            
            # Configurar Tesseract para ser más preciso
            custom_config = r'--oem 3 --psm 11 -c tessedit_char_blacklist=|_-+=<>[]{}()~'
            text_data = pytesseract.image_to_data(img_pil, output_type=pytesseract.Output.DICT, config=custom_config)
            
            # Procesar alturas de texto con filtrado adicional
            text_heights = []
            for i in range(len(text_data['text'])):
                # Solo procesar elementos con texto y confianza alta
                if (text_data['text'][i].strip() and 
                    text_data['conf'][i] > 40 and  # Aumentar umbral de confianza
                    len(text_data['text'][i]) > 1):  # Ignorar caracteres sueltos
                    
                    x, y, w, h = (text_data['left'][i], text_data['top'][i],
                                text_data['width'][i], text_data['height'][i])
                    
                    # Verificar proporción de aspecto y tamaño mínimo
                    if (w > 5 and h > 5 and  # Tamaño mínimo
                        w < img_cv.shape[1] * 0.8 and  # No más del 80% del ancho
                        h < img_cv.shape[0] * 0.3):    # No más del 30% del alto
                        
                        height_mm = self._pixels_to_mm(h)
                        # Solo incluir alturas razonables (entre 0.5mm y 20mm)
                        if 0.5 < height_mm < 20:
                            text_heights.append(height_mm)
            
            return text_heights
            
        except Exception as e:
            print(f"Error en _detect_text_heights: {str(e)}")
            return []
    
    def _draw_detection_boxes(self, image: np.ndarray, logo_box: Tuple[int, int, int, int], text_boxes: List[Dict]) -> np.ndarray:
        """Dibuja cajas de detección en la imagen."""
        # Asegurarse de que la imagen sea un array de numpy válido
        if not isinstance(image, np.ndarray):
            raise ValueError("La imagen debe ser un array de numpy")

        # Hacer una copia de la imagen para no modificar la original
        vis_image = image.copy()
        
        # Convertir la imagen al formato correcto para dibujar
        if len(vis_image.shape) == 2:  # Si es escala de grises
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        elif vis_image.shape[2] == 4:  # Si tiene canal alfa
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGRA2BGR)
        elif vis_image.shape[2] != 3:  # Si no es BGR
            raise ValueError("Formato de imagen no soportado")
        
        # Asegurarse de que la imagen sea uint8
        if vis_image.dtype != np.uint8:
            vis_image = (vis_image * 255).astype(np.uint8)
        
        try:
            # Dibujar caja roja alrededor del logo completo
            x, y, w, h = logo_box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Dibujar cajas azules alrededor de los textos detectados
            for box in text_boxes:
                x, y, w, h = box['box']
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        except Exception as e:
            print(f"Error al dibujar cajas: {str(e)}")
            return image  # Devolver la imagen original si hay error
            
        return vis_image

    def _analyze_size_variant(self, image: np.ndarray, target_size_mm: float) -> Dict[str, Any]:
        """Analiza una variante de tamaño del logo."""
        # Verificar que la imagen sea válida
        if not isinstance(image, np.ndarray):
            raise ValueError("La imagen debe ser un array de numpy")

        # Detectar límites del logo
        x, y, w, h = self._detect_logo_boundaries(image)
        
        # Calcular dimensiones en milímetros
        width_mm = self._pixels_to_mm(w)
        height_mm = self._pixels_to_mm(h)
        
        # Extraer la región del logo
        logo_region = image[y:y+h, x:x+w]
        
        # Detectar alturas de texto
        text_heights = self._detect_text_heights(logo_region)
        
        return {
            'width_mm': width_mm,
            'height_mm': height_mm,
            'text_heights_mm': text_heights,
            'box': (x, y, w, h)
        }

    def analyze_and_resize_logo(self, image_path: str, target_size_mm: float) -> Dict:
        """
        Analiza y redimensiona un logo para un tamaño objetivo.
        
        Args:
            image_path: Ruta al archivo de imagen
            target_size_mm: Tamaño objetivo en milímetros
            
        Returns:
            Dict con los resultados del análisis y las imágenes redimensionadas
        """
        try:
            # Cargar imagen
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            # Asegurar que la imagen tenga el formato correcto
            if len(image.shape) == 2:  # Si es escala de grises
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
            elif len(image.shape) == 3 and image.shape[2] == 3:  # Si es BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            
            # Calcular factores de escala para cada variante
            resultados = {}
            variantes = {
                'chico': target_size_mm - 10,
                'solicitado': target_size_mm,
                'grande': target_size_mm + 10
            }
            
            imagenes = {}
            for nombre, size_mm in variantes.items():
                # Calcular factor de escala
                scale_factor = size_mm / target_size_mm
                
                # Redimensionar imagen
                new_width = int(image.shape[1] * scale_factor)
                new_height = int(image.shape[0] * scale_factor)
                resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                # Analizar variante
                resultados[nombre] = self._analyze_size_variant(resized, size_mm)
                imagenes[nombre] = resized
            
            # Generar respuesta basada en los resultados
            respuesta = self._generar_respuesta(resultados['solicitado'], target_size_mm)
            
            return {
                'resultados': resultados,
                'imagenes': imagenes,
                'respuesta': respuesta
            }
            
        except Exception as e:
            raise ValueError(f"Error al analizar el logo: {str(e)}")

    def _generar_respuesta(self, resultados: Dict, target_size_mm: float) -> str:
        """Genera una respuesta en lenguaje natural basada en los resultados."""
        try:
            # Extraer métricas clave
            width_mm = resultados['width_mm']
            height_mm = resultados['height_mm']
            text_heights = resultados['text_heights_mm']
            
            # Inicializar respuesta
            respuesta = []
            
            # Analizar dimensiones generales
            respuesta.append(f"El logo mide {width_mm:.1f}mm de ancho por {height_mm:.1f}mm de alto.")
            
            # Analizar textos
            if text_heights:
                min_text = min(text_heights)
                avg_text = sum(text_heights) / len(text_heights)
                texts_below_2mm = len([h for h in text_heights if h < 2.0])
                
                if texts_below_2mm > 0:
                    respuesta.append(f"⚠️ Se detectaron {texts_below_2mm} elementos de texto menores a 2mm.")
                    if min_text < 1.0:
                        respuesta.append("❌ Algunos textos son demasiado pequeños y podrían ser ilegibles.")
                    else:
                        respuesta.append("⚠️ Algunos textos podrían ser difíciles de leer.")
                else:
                    respuesta.append("✅ Todos los textos tienen un tamaño adecuado.")
                
                respuesta.append(f"El texto más pequeño mide {min_text:.1f}mm y el promedio es {avg_text:.1f}mm.")
            else:
                respuesta.append("No se detectaron elementos de texto en el logo.")
            
            # Recomendaciones
            if width_mm > target_size_mm * 1.2 or height_mm > target_size_mm * 1.2:
                respuesta.append("📏 El logo es significativamente más grande que el tamaño solicitado.")
            elif width_mm < target_size_mm * 0.8 or height_mm < target_size_mm * 0.8:
                respuesta.append("📏 El logo es significativamente más pequeño que el tamaño solicitado.")
            
            if text_heights and min_text < 2.0:
                respuesta.append("💡 Considera aumentar el tamaño para mejorar la legibilidad de los textos pequeños.")
            
            return " ".join(respuesta)
            
        except Exception as e:
            return f"Error al generar respuesta: {str(e)}"