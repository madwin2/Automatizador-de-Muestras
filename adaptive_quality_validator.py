import cv2
import numpy as np
from typing import Dict, Any, List, Tuple

class AdaptiveQualityValidator:
    def __init__(self):
        self.quality_thresholds = {
            "preservacion_detalles": 0.9,  # 90% mínimo
            "contraste": 0.8,              # 80% mínimo
            "limpieza": 0.95,              # 95% mínimo
            "integridad_bordes": 0.9       # 90% mínimo
        }

    def validate_step(self, original_image: np.ndarray, processed_image: np.ndarray, 
                     step_name: str, parameters: Dict) -> Dict[str, Any]:
        """Valida la calidad de un paso de procesamiento."""
        metrics = {}
        
        # Análisis de preservación de detalles
        detail_score = self._analyze_detail_preservation(original_image, processed_image)
        metrics["preservacion_detalles"] = detail_score
        
        # Análisis de contraste
        contrast_score = self._analyze_contrast(processed_image)
        metrics["contraste"] = contrast_score
        
        # Análisis de limpieza
        noise_score = self._analyze_noise(processed_image)
        metrics["limpieza"] = noise_score
        
        # Análisis de integridad de bordes
        edge_score = self._analyze_edge_integrity(original_image, processed_image)
        metrics["integridad_bordes"] = edge_score
        
        # Análisis de elementos decorativos
        decorative_score = self._analyze_decorative_elements(original_image, processed_image)
        metrics["preservacion_decorativos"] = decorative_score
        
        # Calcular puntuación global
        global_score = self._calculate_global_score(metrics)
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(
            metrics, step_name, parameters, global_score
        )
        
        return {
            "metricas": metrics,
            "puntuacion_global": global_score,
            "recomendaciones": recommendations,
            "paso_valido": global_score >= 0.8  # 80% mínimo para considerar válido
        }

    def _analyze_detail_preservation(self, original: np.ndarray, 
                                   processed: np.ndarray) -> float:
        """Analiza la preservación de detalles entre la imagen original y procesada."""
        if len(original.shape) > 2:
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original
            
        if len(processed.shape) > 2:
            processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            processed_gray = processed

        # Detectar bordes en ambas imágenes
        edges_original = cv2.Canny(original_gray, 50, 150)
        edges_processed = cv2.Canny(processed_gray, 50, 150)
        
        # Calcular coincidencia de bordes
        matched_edges = cv2.bitwise_and(edges_original, edges_processed)
        
        if np.sum(edges_original) == 0:
            return 1.0
            
        return np.sum(matched_edges) / np.sum(edges_original)

    def _analyze_contrast(self, image: np.ndarray) -> float:
        """Analiza el contraste de la imagen."""
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Calcular histograma
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Normalizar histograma
        hist = hist.ravel() / hist.sum()
        
        # Calcular entropía del contraste
        non_zero = hist > 0
        entropy = -np.sum(hist[non_zero] * np.log2(hist[non_zero]))
        
        # Normalizar a [0, 1]
        max_entropy = np.log2(256)
        return min(entropy / max_entropy, 1.0)

    def _analyze_noise(self, image: np.ndarray) -> float:
        """Analiza el nivel de ruido en la imagen."""
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Aplicar filtro de mediana
        denoised = cv2.medianBlur(gray, 3)
        
        # Calcular diferencia con imagen original
        noise = cv2.absdiff(gray, denoised)
        
        # Calcular ratio de ruido
        noise_ratio = 1.0 - (np.sum(noise) / (gray.shape[0] * gray.shape[1] * 255))
        
        return noise_ratio

    def _analyze_edge_integrity(self, original: np.ndarray, 
                              processed: np.ndarray) -> float:
        """Analiza la integridad de los bordes."""
        if len(original.shape) > 2:
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original
            
        if len(processed.shape) > 2:
            processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            processed_gray = processed

        # Detectar contornos en ambas imágenes
        contours_original = cv2.findContours(
            cv2.threshold(original_gray, 127, 255, cv2.THRESH_BINARY)[1],
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        
        contours_processed = cv2.findContours(
            cv2.threshold(processed_gray, 127, 255, cv2.THRESH_BINARY)[1],
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        
        if not contours_original:
            return 1.0
            
        # Comparar áreas de contornos
        area_original = sum(cv2.contourArea(c) for c in contours_original)
        area_processed = sum(cv2.contourArea(c) for c in contours_processed)
        
        if area_original == 0:
            return 1.0
            
        return min(area_processed / area_original, 1.0)

    def _analyze_decorative_elements(self, original: np.ndarray, 
                                    processed: np.ndarray) -> float:
        """Analiza la preservación de elementos decorativos."""
        # Detectar elementos decorativos en ambas imágenes
        decorative_original = self._detect_decorative_elements(original)
        decorative_processed = self._detect_decorative_elements(processed)
        
        if not decorative_original["elementos_decorativos"]:
            return 1.0
            
        # Comparar elementos
        preserved_elements = 0
        total_elements = len(decorative_original["elementos_decorativos"])
        
        for elem_orig in decorative_original["elementos_decorativos"]:
            # Buscar elemento correspondiente en la imagen procesada
            for elem_proc in decorative_processed["elementos_decorativos"]:
                if elem_orig["tipo"] == elem_proc["tipo"]:
                    if elem_orig["tipo"] == "línea":
                        # Comparar longitud y ángulo
                        length_diff = abs(elem_orig["longitud"] - elem_proc["longitud"])
                        angle_diff = abs(elem_orig["angulo"] - elem_proc["angulo"])
                        
                        if length_diff < 10 and angle_diff < 15:
                            preserved_elements += 1
                            break
                    elif elem_orig["tipo"] == "curva":
                        # Comparar área y circularidad
                        area_diff = abs(elem_orig["area"] - elem_proc["area"])
                        circ_diff = abs(elem_orig["circularidad"] - elem_proc["circularidad"])
                        
                        if area_diff / elem_orig["area"] < 0.2 and circ_diff < 0.1:
                            preserved_elements += 1
                            break
        
        return preserved_elements / total_elements if total_elements > 0 else 1.0

    def _calculate_global_score(self, metrics: Dict[str, float]) -> float:
        """Calcula la puntuación global basada en todas las métricas."""
        weights = {
            "preservacion_detalles": 0.3,
            "contraste": 0.15,
            "limpieza": 0.15,
            "integridad_bordes": 0.2,
            "preservacion_decorativos": 0.2  # Peso significativo para elementos decorativos
        }
        
        return sum(metrics[key] * weights[key] for key in weights if key in metrics)

    def _generate_recommendations(self, metrics: Dict[str, float], 
                                step_name: str, parameters: Dict,
                                global_score: float) -> List[str]:
        """Genera recomendaciones basadas en las métricas."""
        recommendations = []
        
        # Verificar cada métrica contra su umbral
        for metric, value in metrics.items():
            if value < self.quality_thresholds.get(metric, 0.8):
                if step_name == "binarizar":
                    if metric == "preservacion_detalles":
                        recommendations.append(
                            f"Ajustar umbral de binarización: "
                            f"{'aumentar' if value < 0.5 else 'disminuir'}"
                        )
                elif step_name == "eliminar_ruido":
                    if metric == "limpieza":
                        recommendations.append(
                            f"Ajustar kernel_size: "
                            f"{'aumentar' if value < 0.5 else 'disminuir'}"
                        )
                elif step_name == "engrosar_trazos":
                    if metric == "integridad_bordes":
                        recommendations.append(
                            f"Ajustar pixels de engrosamiento: "
                            f"{'aumentar' if value < 0.5 else 'disminuir'}"
                        )
        
        return recommendations 

    def _detect_decorative_elements(self, image: np.ndarray) -> Dict[str, Any]:
        """Detecta elementos decorativos como líneas y curvas."""
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detectar líneas usando transformada de Hough probabilística
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=20, maxLineGap=10)

        # Detectar curvas usando contornos y análisis de curvatura
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        decorative_elements = []
        
        # Analizar líneas
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                
                # Analizar si es un elemento decorativo
                if length > 20:  # Líneas significativas
                    decorative_elements.append({
                        "tipo": "línea",
                        "longitud": float(length),
                        "angulo": float(angle),
                        "posicion": [int(x1), int(y1), int(x2), int(y2)]
                    })

        # Analizar curvas
        for contour in contours:
            # Calcular perímetro y área
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            
            if perimeter > 0:
                # Calcular circularidad
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Si no es muy circular (para excluir logos/letras)
                if 0.1 < circularity < 0.7 and area > 100:
                    # Aproximar la curva
                    epsilon = 0.02 * perimeter
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    decorative_elements.append({
                        "tipo": "curva",
                        "perimetro": float(perimeter),
                        "area": float(area),
                        "circularidad": float(circularity),
                        "puntos": len(approx),
                        "posicion": approx.reshape(-1, 2).tolist()
                    })

        return {
            "elementos_decorativos": decorative_elements,
            "total_elementos": len(decorative_elements)
        } 
