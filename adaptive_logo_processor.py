import cv2
import numpy as np
import base64
from logo_processor import LogoProcessor
from logo_analyzer import LogoAnalyzer
from adaptive_quality_validator import AdaptiveQualityValidator
from openai import OpenAI
import os
import json
from typing import Dict, Any, List, Tuple

class AdaptiveLogoProcessor(LogoProcessor):
    def __init__(self, api_key: str):
        super().__init__()
        self.client = OpenAI(api_key=api_key)
        self.analyzer = LogoAnalyzer()
        self.validator = AdaptiveQualityValidator()
        
    def _encode_image(self, image: np.ndarray) -> str:
        """Codifica una imagen numpy en base64."""
        success, buffer = cv2.imencode('.png', image)
        if not success:
            raise ValueError("No se pudo codificar la imagen")
        return base64.b64encode(buffer).decode('utf-8')
    
    def _get_gpt_feedback(self, original_image: np.ndarray, processed_image: np.ndarray, 
                         current_step: Dict, next_step: Dict = None) -> Dict:
        """Obtiene feedback de GPT sobre el resultado del paso actual."""
        
        def analizar_imagen(img):
            if len(img.shape) > 2:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                gray = img
            
            # Detectar texto usando diferentes umbrales
            textos_detectados = []
            for threshold in [127, 150, 100, 200]:  # Añadido umbral alto para texto fino
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 5 and h > 5:  # Reducido el tamaño mínimo para detectar detalles más finos
                        aspect_ratio = float(w) / h
                        if 0.05 < aspect_ratio < 20:  # Ampliado el rango para detectar más formas
                            area = cv2.contourArea(contour)
                            textos_detectados.append({
                                "area": float(area),
                                "aspect_ratio": float(aspect_ratio),
                                "posicion": [int(x), int(y), int(w), int(h)]
                            })
            
            # Detectar bordes con diferentes sensibilidades
            bordes_suaves = cv2.Canny(gray, 30, 100)  # Más sensible a bordes suaves
            bordes_fuertes = cv2.Canny(gray, 100, 200)
            
            return {
                "elementos_detectados": len(textos_detectados),
                "areas_texto": [t["area"] for t in textos_detectados],
                "bordes_suaves": int(np.count_nonzero(bordes_suaves)),
                "bordes_fuertes": int(np.count_nonzero(bordes_fuertes)),
                "contraste": float(np.std(gray)),
                "valor_medio": float(np.mean(gray))
            }
        
        original_stats = analizar_imagen(original_image)
        processed_stats = analizar_imagen(processed_image)
        
        # Calcular pérdida de información
        perdida_elementos = original_stats["elementos_detectados"] - processed_stats["elementos_detectados"]
        perdida_bordes_suaves = (original_stats["bordes_suaves"] - processed_stats["bordes_suaves"]) / max(1, original_stats["bordes_suaves"])
        perdida_bordes_fuertes = (original_stats["bordes_fuertes"] - processed_stats["bordes_fuertes"]) / max(1, original_stats["bordes_fuertes"])
        
        # Calcular calidad automáticamente
        calidad_base = 8  # Comenzamos con una calidad base más alta
        
        # Penalizar más fuertemente la pérdida de elementos
        if perdida_elementos > 0:
            calidad_base -= min(4, perdida_elementos * 1.5)  # Penalización más fuerte por pérdida de elementos
        
        # Penalizar la pérdida de bordes suaves (detalles finos)
        if perdida_bordes_suaves > 0.2:  # Más sensible a la pérdida de detalles suaves
            calidad_base -= 2
        
        # Penalizar la pérdida de bordes fuertes (estructura principal)
        if perdida_bordes_fuertes > 0.3:
            calidad_base -= 2
        
        # Penalizar pérdida de contraste
        if processed_stats["contraste"] < original_stats["contraste"] * 0.6:
            calidad_base -= 2
        
        calidad = max(1, min(10, calidad_base))
        
        # Determinar si necesitamos ajustes
        ajustes = None
        if calidad < 7:
            if current_step["herramienta"] == "binarizar":
                # Ajustar umbral de manera más agresiva
                if perdida_elementos > 0:
                    nuevo_threshold = int(processed_stats["valor_medio"] * 1.2)  # Más agresivo
                else:
                    nuevo_threshold = int(processed_stats["valor_medio"])
                ajustes = {
                    "parametros": {"threshold": nuevo_threshold},
                    "justificacion": f"Ajustando umbral a {nuevo_threshold} para preservar más detalles"
                }
            elif current_step["herramienta"] == "engrosar_trazos":
                # Ser más conservador con el engrosamiento
                pixels = current_step["parametros"].get("pixels", 2)
                if perdida_elementos > 0 or perdida_bordes_suaves > 0.2:
                    pixels = 1  # Reducir al mínimo si hay pérdida de detalles
                    ajustes = {
                        "parametros": {"pixels": pixels},
                        "justificacion": "Reduciendo engrosamiento al mínimo para preservar detalles finos"
                    }
            elif current_step["herramienta"] == "eliminar_ruido":
                # Ajustar el kernel_size según la pérdida de detalles
                if perdida_elementos > 0:
                    ajustes = {
                        "parametros": {"kernel_size": 2},  # Usar kernel más pequeño
                        "justificacion": "Reduciendo tamaño del kernel para preservar más detalles"
                    }
        
        return {
            "calidad_resultado": calidad,
            "ajustes_recomendados": ajustes,
            "continuar_siguiente_paso": calidad >= 6,  # Más estricto con la calidad mínima
            "observaciones": f"Elementos detectados: {processed_stats['elementos_detectados']}, "
                           f"Pérdida de bordes suaves: {perdida_bordes_suaves:.2%}, "
                           f"Pérdida de bordes fuertes: {perdida_bordes_fuertes:.2%}, "
                           f"Contraste: {processed_stats['contraste']:.2f}"
        }

    def _ensure_black_logo_white_background(self, image):
        """Asegura que el logo sea negro y el fondo blanco"""
        if image is None:
            raise ValueError("La imagen es nula")
            
        # Convertir a escala de grises si es necesario
        if len(image.shape) > 2:
            if image.shape[2] == 4:  # BGRA
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:  # BGR
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Binarizar con umbral Otsu
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Verificar el color predominante del fondo (asumimos que los bordes son fondo)
        border_pixels = np.concatenate([
            binary[0, :],  # top border
            binary[-1, :],  # bottom border
            binary[:, 0],  # left border
            binary[:, -1]  # right border
        ])
        
        # Contar píxeles blancos y negros en el borde
        white_pixels = np.sum(border_pixels > 127)
        black_pixels = len(border_pixels) - white_pixels
        
        # Si hay más píxeles negros en el borde, necesitamos invertir
        if black_pixels > white_pixels:
            print("\n🔄 Invirtiendo colores: el fondo debe ser blanco")
            binary = cv2.bitwise_not(binary)
        
        return binary

    def _final_background_check(self, image):
        """Verificación final para asegurar que el fondo sea mayoritariamente blanco"""
        # Contar píxeles negros y blancos en toda la imagen
        total_pixels = image.size
        black_pixels = np.sum(image < 127)
        white_pixels = total_pixels - black_pixels
        
        # Si más del 60% de la imagen es negra, probablemente el fondo es negro
        if black_pixels / total_pixels > 0.6:
            print("\n🔄 Inversión final: detectado exceso de píxeles negros")
            return cv2.bitwise_not(image)
        return image

    def _trim_borders(self, image):
        """Recorta los bordes con un método más robusto."""
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
        else:
            gray = image.copy()

        # Aplicar umbral adaptativo
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Aplicar operaciones morfológicas para limpiar ruido
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Encontrar los bordes del contenido
        coords = cv2.findNonZero(binary)
        if coords is None or len(coords) < 10:
            return image
            
        x, y, w, h = cv2.boundingRect(coords)
        
        # Agregar un pequeño margen
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        
        # Recortar la imagen
        if len(image.shape) > 2:
            return image[y:y+h, x:x+w]
        return gray[y:y+h, x:x+w]

    def _needs_inversion(self, image):
        """Determina si un logo necesita ser invertido usando análisis más sofisticado."""
        try:
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
            else:
                gray = image

            # Analizar histograma
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / gray.size

            # Encontrar los dos picos más significativos
            peaks = []
            for i in range(1, 255):
                if hist[i] > 0.01 and hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                    peaks.append((i, hist[i]))
            
            peaks.sort(key=lambda x: x[1], reverse=True)
            
            if len(peaks) >= 2:
                # Si el pico más alto está en la región oscura y el segundo en la clara
                if peaks[0][0] < 128 and peaks[1][0] > 128:
                    print("🔄 Inversión recomendada basada en análisis de histograma")
                    return True

            # Análisis de bordes
            border = np.concatenate([
                gray[0, :],    # borde superior
                gray[-1, :],   # borde inferior
                gray[:, 0],    # borde izquierdo
                gray[:, -1]    # borde derecho
            ])
            
            # Si los bordes son mayoritariamente oscuros, probablemente necesitamos invertir
            if np.mean(border) < 128:
                print("🔄 Inversión recomendada basada en análisis de bordes")
                return True

            return False
        except Exception as e:
            print(f"Error en _needs_inversion: {str(e)}")
            return False

    def _invert_if_needed(self, image):
        """Invierte la imagen si es necesario."""
        if self._needs_inversion(image):
            return cv2.bitwise_not(image)
        return image

    def _is_optimal_logo(self, image):
        """Determina si un logo ya está en condiciones óptimas."""
        try:
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
            else:
                gray = image.copy()

            # Análisis de histograma
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten()
            
            # Normalizar histograma
            hist = hist / hist.sum()
            
            # Encontrar picos significativos
            peaks = []
            for i in range(1, 255):
                if hist[i] > 0.01 and hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                    peaks.append((i, hist[i]))
            
            # Ordenar picos por altura
            peaks.sort(key=lambda x: x[1], reverse=True)
            
            # Criterios para un logo óptimo
            if len(peaks) >= 2:
                top_peaks = peaks[:2]
                
                # Verificar si los picos están bien separados (uno cerca del negro y otro cerca del blanco)
                dark_peak = min(top_peaks, key=lambda x: x[0])
                light_peak = max(top_peaks, key=lambda x: x[0])
                
                if dark_peak[0] < 50 and light_peak[0] > 200:
                    # Verificar que los picos sean significativos
                    if dark_peak[1] > 0.01 and light_peak[1] > 0.01:
                        # Verificar que no haya muchos valores intermedios
                        mid_range_values = hist[50:200].sum()
                        if mid_range_values < 0.2:  # Menos del 20% de píxeles en el rango medio
                            return True
            
            return False
        except Exception as e:
            print(f"Error en _is_optimal_logo: {str(e)}")
            return False

    def process_logo(self, input_path: str, steps: List[Dict[str, Any]], output_path: str = None) -> Tuple[str, Dict]:
        """
        Procesa un logo aplicando una serie de pasos de manera adaptativa.
        
        Args:
            input_path (str): Ruta al archivo de entrada
            steps (List[Dict[str, Any]]): Lista de pasos a aplicar
            output_path (str, optional): Ruta para guardar el resultado
        
        Returns:
            Tuple[str, Dict]: Ruta del archivo procesado y resultados del procesamiento
        """
        try:
            # Leer la imagen
            image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"No se pudo leer la imagen: {input_path}")
            
            # Verificar si el logo ya está en condiciones óptimas
            if self._is_optimal_logo(image):
                print("\n✨ Logo detectado en condiciones óptimas. No se requiere procesamiento.")
                if output_path:
                    cv2.imwrite(output_path, image)
                    return output_path, {"usar_original": True}
                else:
                    temp_output = "temp/no_processing_needed.png"
                    os.makedirs("temp", exist_ok=True)
                    cv2.imwrite(temp_output, image)
                    return temp_output, {"usar_original": True}
            
            # Preparar la imagen inicial
            processed = self._ensure_black_logo_white_background(image)
            
            # Lista para almacenar los resultados de cada paso
            resultados_pasos = []
            
            # Procesar cada paso
            for i, step in enumerate(steps):
                try:
                    # Guardar la imagen antes del paso
                    original_before_step = processed.copy()
                    
                    # Aplicar el paso actual
                    herramienta = step["herramienta"]
                    parametros = step.get("parametros", {})
                    
                    # Procesar según la herramienta
                    if hasattr(self, herramienta):
                        processed = getattr(self, herramienta)(processed, **parametros)
                    else:
                        print(f"\n⚠️ Herramienta no encontrada: {herramienta}")
                        continue
                    
                    # Obtener feedback sobre el resultado
                    next_step = steps[i + 1] if i + 1 < len(steps) else None
                    feedback = self._get_gpt_feedback(original_before_step, processed, step, next_step)
                    
                    # Verificar si necesitamos ajustar parámetros
                    if feedback["ajustes_recomendados"]:
                        ajustes = feedback["ajustes_recomendados"]
                        print(f"\n🔧 Ajustando parámetros: {ajustes['justificacion']}")
                        
                        # Aplicar el paso con los parámetros ajustados
                        processed = getattr(self, herramienta)(
                            original_before_step,
                            **ajustes["parametros"]
                        )
                        
                        # Actualizar feedback con los nuevos parámetros
                        feedback = self._get_gpt_feedback(
                            original_before_step, processed,
                            {**step, "parametros": ajustes["parametros"]},
                            next_step
                        )
                    
                    # Registrar resultados del paso
                    resultados_pasos.append({
                        "herramienta": herramienta,
                        "parametros_originales": parametros,
                        "parametros_ajustados": feedback["ajustes_recomendados"]["parametros"] if feedback["ajustes_recomendados"] else None,
                        "calidad": feedback["calidad_resultado"],
                        "observaciones": feedback["observaciones"]
                    })
                    
                    # Si la calidad no es suficiente y no podemos mejorarla, considerar usar el original
                    if feedback["calidad_resultado"] < 5 and not feedback["ajustes_recomendados"]:
                        print("\n⚠️ Calidad insuficiente y no hay ajustes disponibles")
                        if self._is_optimal_logo(image):
                            print("✨ Usando imagen original que ya estaba en condiciones óptimas")
                            processed = image
                            break
                    
                    # Verificar si debemos continuar con el siguiente paso
                    if not feedback["continuar_siguiente_paso"]:
                        print("\n⚠️ Calidad insuficiente para continuar con el siguiente paso")
                        break
                        
                except Exception as e:
                    print(f"\n❌ Error en paso {herramienta}: {str(e)}")
                    resultados_pasos.append({
                        "herramienta": herramienta,
                        "error": str(e)
                    })
            
            # Verificación final del fondo
            processed = self._final_background_check(processed)
            
            # Guardar resultado
            if output_path:
                cv2.imwrite(output_path, processed)
                final_path = output_path
            else:
                temp_output = "temp/processed_logo.png"
                os.makedirs("temp", exist_ok=True)
                cv2.imwrite(temp_output, processed)
                final_path = temp_output
            
            return final_path, {
                "pasos": resultados_pasos,
                "usar_original": False
            }
            
        except Exception as e:
            print(f"\n❌ Error en process_logo: {str(e)}")
            raise

    def _adjust_parameters(self, herramienta: str, 
                         parametros: Dict, 
                         recomendaciones: List[str]) -> Dict:
        """Ajusta los parámetros basándose en las recomendaciones."""
        try:
            if herramienta == "binarizar":
                if "aumentar_contraste" in recomendaciones:
                    parametros["threshold"] = min(255, parametros.get("threshold", 128) + 20)
                elif "reducir_contraste" in recomendaciones:
                    parametros["threshold"] = max(0, parametros.get("threshold", 128) - 20)
                    
            elif herramienta == "eliminar_ruido":
                if "preservar_detalles" in recomendaciones:
                    parametros["kernel_size"] = max(1, parametros.get("kernel_size", 3) - 2)
                elif "eliminar_mas_ruido" in recomendaciones:
                    parametros["kernel_size"] = min(7, parametros.get("kernel_size", 3) + 2)
                    
            elif herramienta == "engrosar_trazos":
                if "reducir_engrosamiento" in recomendaciones:
                    parametros["pixels"] = max(1, parametros.get("pixels", 2) - 1)
                elif "aumentar_engrosamiento" in recomendaciones:
                    parametros["pixels"] = min(5, parametros.get("pixels", 2) + 1)
            
            return parametros
        except Exception as e:
            print(f"Error en _adjust_parameters: {str(e)}")
            return parametros