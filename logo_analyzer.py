import cv2
import numpy as np
import pytesseract
from PIL import Image
from skimage import feature, color, segmentation, filters
import os
from typing import Dict, Any, Tuple
import sys

class LogoAnalyzer:
    def __init__(self):
        # Configurar pytesseract con la ruta específica
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def analyze_logo(self, image_path: str) -> Dict[str, Any]:
        """
        Analiza un logo y retorna información detallada sobre el mismo.
        
        Args:
            image_path: Ruta a la imagen del logo
            
        Returns:
            Dict con información sobre el logo incluyendo:
            - dimensiones
            - textos encontrados
            - colores dominantes
            - información sobre el fondo
            - formas detectadas
            - elementos decorativos
        """
        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Convertir a RGB para procesamiento
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detectar el área del logo
        logo_mask = self._detect_logo_area(image_rgb)
        logo_bbox = self._get_bounding_box(logo_mask)
        
        # Recortar la imagen al área del logo
        x, y, w, h = logo_bbox
        logo_area = image_rgb[y:y+h, x:x+w]
        
        # Obtener dimensiones del logo
        height, width = logo_area.shape[:2]
        
        # Detectar elementos decorativos
        elementos_decorativos = self._detect_decorative_elements(image)
        
        # Analizar componentes
        results = {
            "dimensiones_totales": {
                "ancho": image.shape[1],
                "alto": image.shape[0]
            },
            "dimensiones_logo": {
                "ancho": width,
                "alto": height,
                "relacion_aspecto": width/height if height > 0 else 0,
                "posicion_x": x,
                "posicion_y": y
            },
            "textos": self._extract_text(image_rgb),
            "colores": self._analyze_colors(logo_area),
            "fondo": self._analyze_background(logo_area),
            "formas": self._detect_shapes(logo_area),
            "elementos_decorativos": elementos_decorativos
        }
        
        return results

    def _detect_logo_area(self, image: np.ndarray) -> np.ndarray:
        """Detecta el área que contiene el logo."""
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Aplicar umbral adaptativo
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Encontrar contornos
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Crear máscara
        mask = np.zeros_like(gray)
        
        # Dibujar contornos significativos
        min_area = image.shape[0] * image.shape[1] * 0.001  # 0.1% del área total
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(mask, [cnt], -1, 255, -1)
        
        # Operaciones morfológicas para limpiar la máscara
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask

    def _get_bounding_box(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Obtiene el rectángulo delimitador del logo."""
        # Encontrar contornos en la máscara
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return (0, 0, mask.shape[1], mask.shape[0])
        
        # Encontrar el contorno más grande
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Añadir un pequeño margen
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(mask.shape[1] - x, w + 2*margin)
        h = min(mask.shape[0] - y, h + 2*margin)
        
        return (x, y, w, h)

    def _extract_text(self, image: np.ndarray) -> list:
        """Extrae texto de la imagen usando OCR con configuración mejorada."""
        # Preprocesar imagen para mejor detección de texto
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Aplicar umbral adaptativo
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Configurar opciones de OCR
        custom_config = r'--oem 3 --psm 6'
        
        # Detectar texto en la imagen original y en la procesada
        text_original = pytesseract.image_to_string(image, config=custom_config)
        text_thresh = pytesseract.image_to_string(thresh, config=custom_config)
        
        # Combinar resultados y limpiar
        all_text = text_original + "\n" + text_thresh
        words = []
        for line in all_text.split('\n'):
            line_words = [word.strip() for word in line.split() if word.strip()]
            words.extend(line_words)
        
        # Eliminar duplicados y palabras muy cortas
        words = list(set([word for word in words if len(word) > 1]))
        return words

    def _analyze_colors(self, image: np.ndarray) -> Dict[str, Any]:
        """Analiza los colores dominantes en la imagen."""
        # Convertir a formato compatible con k-means
        pixels = image.reshape(-1, 3)
        
        # Usar k-means para encontrar los colores dominantes
        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, labels, centers = cv2.kmeans(
            np.float32(pixels), n_colors, None, criteria, 10, 
            cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Convertir los centros a RGB
        colors = centers.astype(int).tolist()
        
        # Calcular los porcentajes de cada color
        unique, counts = np.unique(labels, return_counts=True)
        percentages = (counts / len(labels) * 100).tolist()
        
        # Filtrar colores con porcentaje significativo (>1%)
        significant_colors = [
            (color, pct) for color, pct in zip(colors, percentages) 
            if pct > 1.0
        ]
        
        return {
            "colores_dominantes": [color for color, _ in significant_colors],
            "porcentajes": [pct for _, pct in significant_colors]
        }

    def _analyze_background(self, image: np.ndarray) -> Dict[str, Any]:
        """Analiza el fondo de la imagen."""
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detectar gradientes
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Analizar variación en el fondo
        threshold = np.mean(gradient_magnitude)
        
        background_type = "sólido"
        if np.max(gradient_magnitude) > threshold * 3:
            background_type = "gradiente o textura"
        
        return {
            "tipo": background_type,
            "variacion_gradiente": float(np.mean(gradient_magnitude))
        }

    def _detect_shapes(self, image: np.ndarray) -> Dict[str, Any]:
        """Detecta formas en la imagen."""
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detectar bordes
        edges = feature.canny(gray, sigma=3)
        
        # Detectar contornos
        contours = cv2.findContours(
            edges.astype(np.uint8), cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )[0]
        
        shapes = []
        for contour in contours:
            # Aproximar el contorno
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Clasificar forma según número de vértices
            vertices = len(approx)
            area = cv2.contourArea(contour)
            
            if area > 100:  # Filtrar formas muy pequeñas
                if vertices == 3:
                    shape_type = "triángulo"
                elif vertices == 4:
                    shape_type = "rectángulo"
                elif vertices == 5:
                    shape_type = "pentágono"
                elif vertices > 5:
                    shape_type = "círculo"
                else:
                    shape_type = "otro"
                
                shapes.append({
                    "tipo": shape_type,
                    "area": float(area),
                    "vertices": int(vertices)
                })
        
        return {
            "formas_detectadas": shapes,
            "total_formas": len(shapes)
        }

    def _detect_decorative_elements(self, image: np.ndarray) -> Dict[str, Any]:
        """Detecta elementos decorativos como líneas, curvas y elementos de diseño."""
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detectar bordes con umbral adaptativo para mejor sensibilidad
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        
        # Mejorar detección de líneas
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                               minLineLength=15, maxLineGap=5)
        
        # Detectar contornos con mejor preservación de detalles
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                     cv2.CHAIN_APPROX_TC89_KCOS)
        
        decorative_elements = []
        
        # Analizar líneas basándose en contexto geométrico
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                
                # Análisis contextual mejorado
                is_decorative = False
                
                # 1. Verificar aislamiento (elemento decorativo vs estructural)
                region = gray[max(0, y1-5):min(gray.shape[0], y1+6),
                            max(0, x1-5):min(gray.shape[1], x1+6)]
                isolation_score = 1.0 - (np.sum(region > 127) / region.size)
                
                # 2. Analizar orientación relativa
                relative_angle = angle % 45  # Ángulo relativo a las orientaciones principales
                angle_score = min(relative_angle, 45 - relative_angle) / 22.5
                
                # 3. Analizar proporción y posición
                relative_length = length / max(image.shape)
                position_score = min(y1, y2) / image.shape[0]  # Favorece elementos en la parte superior
                
                # Combinar factores para decisión
                if (isolation_score > 0.7 or  # Elemento bastante aislado
                    (angle_score > 0.6 and relative_length < 0.3) or  # Ángulo inusual y longitud moderada
                    (position_score > 0.7 and relative_length < 0.2)):  # Elemento superior pequeño
                    is_decorative = True
                
                if is_decorative:
                    decorative_elements.append({
                        "tipo": "línea",
                        "longitud": float(length),
                        "angulo": float(angle),
                        "aislamiento": float(isolation_score),
                        "angulo_relativo": float(angle_score),
                        "posicion_relativa": float(position_score),
                        "posicion": [int(x1), int(y1), int(x2), int(y2)]
                    })

        # Analizar curvas con detección mejorada
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            
            if perimeter > 0 and area > 50:  # Filtro inicial de tamaño
                # Calcular métricas de forma
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                curve_points = contour.reshape(-1, 2)
                
                if len(curve_points) >= 3:
                    # Calcular complejidad de la curva
                    angles = []
                    curvature_variations = []
                    
                    for i in range(len(curve_points)-2):
                        pt1, pt2, pt3 = curve_points[i:i+3]
                        v1 = pt1 - pt2
                        v2 = pt3 - pt2
                        
                        # Ángulo entre segmentos consecutivos
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                        angles.append(angle)
                        
                        if i > 0:
                            # Variación de curvatura
                            curvature_variations.append(abs(angles[-1] - angles[-2]))
                    
                    mean_curvature = np.mean(angles) if angles else 0
                    curvature_variance = np.std(angles) if angles else 0
                    complexity_score = np.mean(curvature_variations) if curvature_variations else 0
                    
                    # Evaluar si es un elemento decorativo basado en características geométricas
                    is_decorative = (
                        (0.1 < circularity < 0.7 and mean_curvature > 0.5) or  # Forma irregular con curvatura
                        (complexity_score > 0.3 and curvature_variance > 0.2) or  # Patrón complejo
                        (len(curve_points) > 10 and curvature_variance > 0.4)  # Curva larga con variaciones
                    )
                    
                    if is_decorative:
                        decorative_elements.append({
                            "tipo": "curva",
                            "perimetro": float(perimeter),
                            "area": float(area),
                            "circularidad": float(circularity),
                            "curvatura_media": float(mean_curvature),
                            "variacion_curvatura": float(curvature_variance),
                            "complejidad": float(complexity_score),
                            "puntos": len(curve_points),
                            "posicion": curve_points.tolist()
                        })

        return {
            "elementos_decorativos": decorative_elements,
            "total_elementos": len(decorative_elements),
            "metricas_globales": {
                "densidad_elementos": len(decorative_elements) / (image.shape[0] * image.shape[1]),
                "complejidad_promedio": np.mean([elem.get("complejidad", 0) for elem in decorative_elements]) if decorative_elements else 0
            }
        }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python logo_analyzer.py <ruta_imagen>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    analyzer = LogoAnalyzer()
    
    try:
        results = analyzer.analyze_logo(image_path)
        print("\nAnálisis del logo:")
        for key, value in results.items():
            print(f"\n{key.upper()}:")
            print(value)
    except Exception as e:
        print(f"Error al analizar el logo: {str(e)}") 
