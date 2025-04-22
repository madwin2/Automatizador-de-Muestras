import cv2
import numpy as np
from typing import Tuple, Dict, Any, List
import logging

class LogoBackgroundAnalyzer:
    def __init__(self):
        """
        Inicializa el analizador de fondos de logos.
        """
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Umbrales y parámetros
        self.BRIGHTNESS_THRESHOLD = 127
        self.CONTRAST_THRESHOLD = 50
        self.NOISE_THRESHOLD = 30
        self.EDGE_THRESHOLD = 100
        self.MIN_CONTOUR_AREA = 100

    def analyze_background(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analiza el fondo de un logo y devuelve un diccionario con los resultados.
        
        Args:
            image: Imagen del logo en formato numpy array
            
        Returns:
            Dict con los resultados del análisis incluyendo:
            - background_type: 'solid', 'transparent', 'complex'
            - background_color: [R,G,B] para fondos sólidos
            - transparency: porcentaje de transparencia
            - complexity_score: puntuación de complejidad
            - recommendations: lista de recomendaciones
        """
        try:
            # Convertir a RGBA si es necesario
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            
            # Extraer canal alfa si existe
            has_alpha = image.shape[2] == 4
            alpha = image[:,:,3] if has_alpha else None
            
            # Analizar transparencia
            transparency_score = self._analyze_transparency(alpha) if has_alpha else 0
            
            # Analizar color de fondo
            background_color = self._detect_background_color(image)
            
            # Analizar complejidad
            complexity_score = self._analyze_complexity(image)
            
            # Determinar tipo de fondo
            background_type = self._determine_background_type(
                transparency_score, 
                complexity_score,
                background_color
            )
            
            # Generar recomendaciones
            recommendations = self._generate_recommendations(
                background_type,
                transparency_score,
                complexity_score,
                background_color
            )
            
            return {
                'background_type': background_type,
                'background_color': background_color,
                'transparency': transparency_score,
                'complexity_score': complexity_score,
                'recommendations': recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Error al analizar el fondo: {str(e)}")
            raise

    def _analyze_transparency(self, alpha: np.ndarray) -> float:
        """
        Analiza el canal alfa para determinar el nivel de transparencia.
        
        Args:
            alpha: Canal alfa de la imagen
            
        Returns:
            float: Porcentaje de transparencia (0-100)
        """
        if alpha is None:
            return 0.0
            
        total_pixels = alpha.size
        transparent_pixels = np.sum(alpha < 255)
        return (transparent_pixels / total_pixels) * 100

    def _detect_background_color(self, image: np.ndarray) -> List[int]:
        """
        Detecta el color de fondo predominante.
        
        Args:
            image: Imagen en formato RGBA
            
        Returns:
            List[int]: Color RGB del fondo
        """
        # Convertir a RGB si es RGBA
        if image.shape[2] == 4:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            image_rgb = image[:,:,:3]
        
        # Aplanar la imagen y contar colores únicos
        pixels = image_rgb.reshape(-1, 3)
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        
        # Encontrar el color más frecuente
        background_color = unique_colors[np.argmax(counts)]
        return background_color.tolist()

    def _analyze_complexity(self, image: np.ndarray) -> float:
        """
        Analiza la complejidad del fondo basado en varios factores.
        
        Args:
            image: Imagen en formato RGBA
            
        Returns:
            float: Puntuación de complejidad (0-100)
        """
        # Convertir a escala de grises
        if image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calcular bordes
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calcular variación de color
        if image.shape[2] == 4:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            image_rgb = image[:,:,:3]
        color_std = np.std(image_rgb)
        
        # Calcular textura usando LBP o varianza local
        texture_score = np.std(cv2.GaussianBlur(gray, (7,7), 0))
        
        # Combinar métricas
        complexity = (
            0.4 * edge_density * 100 +
            0.3 * min(color_std, 100) +
            0.3 * min(texture_score, 100)
        )
        
        return min(complexity, 100)

    def _determine_background_type(
        self,
        transparency_score: float,
        complexity_score: float,
        background_color: List[int]
    ) -> str:
        """
        Determina el tipo de fondo basado en los análisis previos.
        
        Args:
            transparency_score: Porcentaje de transparencia
            complexity_score: Puntuación de complejidad
            background_color: Color RGB del fondo
            
        Returns:
            str: Tipo de fondo ('solid', 'transparent', o 'complex')
        """
        if transparency_score > 50:
            return 'transparent'
        elif complexity_score > 60:
            return 'complex'
        else:
            return 'solid'

    def _generate_recommendations(
        self,
        background_type: str,
        transparency_score: float,
        complexity_score: float,
        background_color: List[int]
    ) -> List[str]:
        """
        Genera recomendaciones basadas en el análisis del fondo.
        
        Args:
            background_type: Tipo de fondo detectado
            transparency_score: Porcentaje de transparencia
            complexity_score: Puntuación de complejidad
            background_color: Color RGB del fondo
            
        Returns:
            List[str]: Lista de recomendaciones
        """
        recommendations = []
        
        if background_type == 'transparent':
            if transparency_score < 90:
                recommendations.append(
                    "Considerar aumentar la transparencia del fondo para mejor integración"
                )
            recommendations.append(
                "El fondo transparente es ideal para superposición en diferentes superficies"
            )
            
        elif background_type == 'complex':
            recommendations.append(
                "Considerar simplificar el fondo para mejor legibilidad"
            )
            if complexity_score > 80:
                recommendations.append(
                    "El fondo muy complejo puede dificultar la visibilidad del logo"
                )
                
        else:  # solid
            brightness = sum(background_color) / 3
            if brightness > 240:
                recommendations.append(
                    "El fondo muy claro puede dificultar la visibilidad en superficies claras"
                )
            elif brightness < 15:
                recommendations.append(
                    "El fondo muy oscuro puede dificultar la visibilidad en superficies oscuras"
                )
                
        return recommendations

    def get_background_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Genera una máscara del fondo del logo.
        
        Args:
            image: Imagen en formato RGBA o RGB
            
        Returns:
            np.ndarray: Máscara binaria donde 255 representa el fondo
        """
        try:
            # Convertir a escala de grises
            if len(image.shape) == 2:
                gray = image
            elif image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Aplicar umbral adaptativo
            thresh = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2
            )
            
            # Encontrar contornos
            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Crear máscara
            mask = np.zeros_like(gray)
            
            # Dibujar contornos grandes
            for cnt in contours:
                if cv2.contourArea(cnt) > self.MIN_CONTOUR_AREA:
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
            
            return mask
            
        except Exception as e:
            self.logger.error(f"Error al generar máscara de fondo: {str(e)}")
            raise

    def suggest_background_removal(self, image: np.ndarray) -> bool:
        """
        Sugiere si el fondo debe ser removido basado en el análisis.
        
        Args:
            image: Imagen en formato RGBA o RGB
            
        Returns:
            bool: True si se recomienda remover el fondo
        """
        try:
            analysis = self.analyze_background(image)
            
            # Criterios para sugerir remoción
            should_remove = (
                analysis['background_type'] == 'complex' or
                (analysis['background_type'] == 'solid' and
                 analysis['complexity_score'] > 30) or
                (analysis['transparency'] > 0 and
                 analysis['transparency'] < 90)
            )
            
            return should_remove
            
        except Exception as e:
            self.logger.error(
                f"Error al sugerir remoción de fondo: {str(e)}"
            )
            raise

    def get_background_stats(self, image: np.ndarray) -> Dict[str, float]:
        """
        Obtiene estadísticas detalladas del fondo.
        
        Args:
            image: Imagen en formato RGBA o RGB
            
        Returns:
            Dict con estadísticas del fondo incluyendo:
            - uniformity: uniformidad del color/textura
            - brightness: brillo promedio
            - contrast: contraste
            - noise: nivel de ruido
        """
        try:
            # Convertir a escala de grises
            if len(image.shape) == 2:
                gray = image
            elif image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calcular estadísticas
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # Calcular ruido usando la desviación estándar local
            noise = cv2.GaussianBlur(gray, (7,7), 0)
            noise = np.std(gray - noise)
            
            # Calcular uniformidad usando histograma
            hist = cv2.calcHist([gray], [0], None, [256], [0,256])
            hist = hist.flatten() / hist.sum()
            uniformity = np.sum(hist ** 2)
            
            return {
                'uniformity': float(uniformity * 100),
                'brightness': float(mean_brightness),
                'contrast': float(std_brightness),
                'noise': float(noise)
            }
            
        except Exception as e:
            self.logger.error(f"Error al obtener estadísticas: {str(e)}")
            raise