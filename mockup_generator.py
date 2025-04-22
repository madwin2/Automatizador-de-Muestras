import cv2
import numpy as np
import os
import subprocess
import sys
import shutil
import time

class MockupGenerator:
    def __init__(self):
        # Definir las rutas base para los scripts y recursos
        self.base_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Creador de Mockup Cuero'))
        self.script_paths = {
            'cuero': os.path.normpath(os.path.join(self.base_path, 'cuero.py')),
            'madera': os.path.normpath(os.path.join(self.base_path, 'madera.py'))
        }
        
        # Rutas de texturas
        self.texture_paths = {
            'cuero': {
                'texture': os.path.normpath(os.path.join(self.base_path, 'cuero2.png'))
            },
            'madera': {
                'texture': os.path.normpath(os.path.join(self.base_path, 'texture_path.png')),
                'burned': os.path.normpath(os.path.join(self.base_path, 'burned_texture_path.png'))
            }
        }
        
        # Verificar que los archivos existen
        for tipo, path in self.script_paths.items():
            if not os.path.exists(path):
                print(f"⚠️ Advertencia: No se encuentra el script de mockup {tipo} en: {path}")
        
        for tipo, paths in self.texture_paths.items():
            for key, path in paths.items():
                if not os.path.exists(path):
                    print(f"⚠️ Advertencia: No se encuentra la textura {key} de {tipo} en: {path}")

    def _escape_path(self, path):
        """Escapa una ruta para que sea segura en strings Python."""
        return path.replace('\\', '\\\\')

    def generate(self, logo_path: str, tipo: str) -> str:
        """
        Genera un mockup usando los scripts de automatización existentes.
        
        Args:
            logo_path: Ruta al archivo del logo procesado
            tipo: Tipo de mockup ('cuero' o 'madera')
            
        Returns:
            str: Ruta del archivo de mockup generado
        """
        if tipo not in self.script_paths:
            raise ValueError(f"Tipo de mockup no válido: {tipo}")
            
        # Crear directorio temporal si no existe
        os.makedirs('temp', exist_ok=True)
        
        # Generar nombres únicos para los archivos temporales
        timestamp = str(int(time.time()))
        temp_logo_name = f'temp_logo_{timestamp}.png'
        temp_output_name = f'temp_output_{timestamp}.png'
        
        # Copiar el logo con un nombre temporal
        logo_copy_path = os.path.normpath(os.path.join(self.base_path, temp_logo_name))
        shutil.copy2(logo_path, logo_copy_path)
        
        try:
            # Definir la ruta de salida
            output_copy_path = os.path.normpath(os.path.join(self.base_path, temp_output_name))
            output_path = os.path.normpath(os.path.join('temp', f'mockup_{tipo}.png'))
            
            # Crear un script temporal nuevo
            temp_script = os.path.normpath(os.path.join(self.base_path, f'temp_{tipo}_{timestamp}.py'))
            
            # Escapar todas las rutas
            texture_path = self._escape_path(self.texture_paths[tipo]['texture'])
            logo_path_escaped = self._escape_path(logo_copy_path)
            output_path_escaped = self._escape_path(output_copy_path)
            
            if tipo == 'cuero':
                script_content = f'''import cv2
import numpy as np
import os
import sys

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cuero import simulate_realistic_leather_embossing

texture_path = "{texture_path}"
logo_path = "{logo_path_escaped}"
output_path = "{output_path_escaped}"

simulate_realistic_leather_embossing(
    texture_path=texture_path,
    logo_path=logo_path,
    output_path=output_path
)
'''
            else:  # madera
                burned_path = self._escape_path(self.texture_paths['madera']['burned'])
                script_content = f'''import cv2
import numpy as np
import os
import sys

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from madera import generate_woodburn_extended_two_halos

texture_path = "{texture_path}"
logo_path = "{logo_path_escaped}"
burned_texture_path = "{burned_path}"
output_path = "{output_path_escaped}"

generate_woodburn_extended_two_halos(
    texture_path=texture_path,
    logo_path=logo_path,
    burned_texture_path=burned_texture_path,
    output_path=output_path
)
'''
            
            # Escribir el script temporal
            with open(temp_script, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            # Ejecutar el script temporal
            result = subprocess.run(
                [sys.executable, temp_script],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Eliminar el script temporal
            os.remove(temp_script)
            
            # Copiar el resultado a la carpeta temp
            if os.path.exists(output_copy_path):
                shutil.move(output_copy_path, output_path)
            
            # Limpiar archivos temporales
            if os.path.exists(logo_copy_path):
                os.remove(logo_copy_path)
            
            # Verificar si el archivo de salida existe
            if not os.path.exists(output_path):
                raise ValueError(f"El script no generó el archivo de salida esperado: {output_path}")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            # Limpiar archivos temporales en caso de error
            if os.path.exists(logo_copy_path):
                os.remove(logo_copy_path)
            if os.path.exists(temp_script):
                os.remove(temp_script)
            raise ValueError(f"Error al ejecutar el script de mockup: {e.stderr}")
        except Exception as e:
            # Limpiar archivos temporales en caso de error
            if os.path.exists(logo_copy_path):
                os.remove(logo_copy_path)
            if os.path.exists(temp_script):
                os.remove(temp_script)
            raise ValueError(f"Error al generar el mockup: {str(e)}")

    def _ajustar_logo(self, logo: np.ndarray, ancho_objetivo: int, alto_objetivo: int) -> np.ndarray:
        """Ajusta el tamaño del logo manteniendo su proporción."""
        h, w = logo.shape[:2]
        ratio = min(ancho_objetivo/w, alto_objetivo/h)
        nuevo_ancho = int(w * ratio)
        nuevo_alto = int(h * ratio)
        return cv2.resize(logo, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)

    def _aplicar_efectos(self, logo: np.ndarray, tipo: str) -> np.ndarray:
        """Aplica efectos específicos según el tipo de mockup."""
        if tipo == 'cuero':
            # Aplicar efecto de relieve
            kernel = np.ones((3,3), np.uint8)
            logo = cv2.erode(logo, kernel, iterations=1)
            logo = cv2.GaussianBlur(logo, (3,3), 0)
        elif tipo == 'madera':
            # Aplicar efecto de grabado
            logo = cv2.GaussianBlur(logo, (3,3), 0)
            logo = cv2.addWeighted(logo, 1.2, logo, 0, 0)
        
        return logo