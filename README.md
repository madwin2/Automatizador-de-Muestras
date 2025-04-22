# Automatizador de Muestras

Este proyecto es una aplicación web Flask que automatiza el procesamiento de logos y la generación de mockups para PhotoRoom.

## Características

- Procesamiento adaptativo de logos
- Análisis de tamaño de logos
- Generación de mockups
- Análisis de fondo de imágenes
- Interfaz web intuitiva

## Requisitos

- Python 3.8 o superior
- OpenAI API Key
- Las dependencias listadas en `requirements.txt`

## Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/madwin2/Automatizador-de-Muestras.git
cd Automatizador-de-Muestras
```

2. Crea un entorno virtual e instala las dependencias:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configura tu API key de OpenAI:
   - Abre el archivo `app.py`
   - Reemplaza `"YOUR_API_KEY_HERE"` con tu API key de OpenAI

4. Crea las carpetas necesarias:
```bash
mkdir uploads temp mockups
```

## Uso

1. Inicia el servidor:
```bash
python app.py
```

2. Abre tu navegador y visita `http://localhost:5000`

3. Utiliza la interfaz web para:
   - Procesar logos
   - Analizar tamaños
   - Generar mockups

## Estructura del Proyecto

- `app.py`: Aplicación principal Flask
- `adaptive_logo_processor.py`: Procesador adaptativo de logos
- `uploads/`: Carpeta para archivos subidos
- `temp/`: Carpeta para archivos temporales
- `mockups/`: Carpeta para mockups generados

## Contribuir

1. Haz fork del repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. Haz commit de tus cambios (`git commit -am 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crea un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.