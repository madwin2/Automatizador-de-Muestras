from flask import Flask, request, render_template, jsonify, send_file, url_for
import os
from werkzeug.utils import secure_filename
from adaptive_logo_processor import AdaptiveLogoProcessor
from mockup_generator import MockupGenerator
from logo_size_analyzer import LogoSizeAnalyzer
from logo_background_analyzer import analyze_and_process_logo
import json
import base64
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# API key de OpenAI
api_key = "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"

# Asegurar que existan las carpetas necesarias
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('temp', exist_ok=True)
os.makedirs('mockups', exist_ok=True)

# Configurar los procesadores
processor = AdaptiveLogoProcessor(api_key)
mockup_generator = MockupGenerator()
logo_size_analyzer = LogoSizeAnalyzer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/size_analyzer')
def size_analyzer_page():
    return render_template('size_analyzer.html')

@app.route('/process', methods=['POST'])
def process_logo():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            # Guardar archivo
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)
            
            # Analizar y procesar el fondo y el logo
            processed_path, preanalysis_result = analyze_and_process_logo(input_path, api_key)
            
            print(f"Resultado del preanálisis: {preanalysis_result}")
            print(f"Ruta de la imagen procesada: {processed_path}")
            
            # Cargar configuración
            with open('pasos_procesamiento.json', 'r') as f:
                config = json.load(f)
            
            # Procesar logo con el resultado del preanálisis
            output_path, resultados = processor.process_logo(processed_path, config['pasos'])
            
            print(f"Ruta de la imagen final: {output_path}")
            
            # Si el logo no necesita ajustes, usar_original será True
            usar_original = resultados.get('usar_original', False)
            
            # Determinar qué imagen usar para la respuesta
            if usar_original and preanalysis_result['is_complex_background'] == 0:
                # Si no hubo procesamiento de PhotoRoom y no necesita ajustes
                with open(input_path, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
            else:
                # Usar la imagen procesada (sea por PhotoRoom o por el proceso normal)
                with open(output_path, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Preparar los pasos
            pasos_procesamiento = resultados.get('pasos', [])
            
            # Agregar el paso de preanálisis al inicio
            pasos_completos = [{
                'nombre': 'Análisis de fondo',
                'id': 'analisis_fondo',
                'mensaje': f"{'Fondo complejo detectado. Enviado a PhotoRoom.' if preanalysis_result['is_complex_background'] == 1 else 'Fondo no complejo. No se envió a PhotoRoom.'} (Complejidad: {preanalysis_result['texture_complexity']})",
                'puntuacion': preanalysis_result['is_complex_background'] * 100,
                'parametros': preanalysis_result
            }]
            
            # Agregar los demás pasos
            if isinstance(pasos_procesamiento, list):
                pasos_completos.extend(pasos_procesamiento)
                # Filtrar cualquier paso inválido o no dict
                pasos_completos = [
                    paso for paso in pasos_completos
                    if isinstance(paso, dict) and ('herramienta' in paso or paso.get('id') == 'analisis_fondo')
                ]
            
            # Devolver resultados
            return jsonify({
                'success': 1,
                'processed_image': f'data:image/png;base64,{img_data}',
                'results': {
                    'pasos': pasos_completos,
                    'usar_original': int(usar_original)
                }
            })
            
        except Exception as e:
            print(f"Error en el procesamiento: {str(e)}")
            return jsonify({'error': str(e)}), 500

@app.route('/apply_mockup', methods=['POST'])
def apply_mockup():
    try:
        data = request.get_json()
        if not data or 'processed_image' not in data or 'tipo' not in data:
            return jsonify({'error': 'Datos incompletos'}), 400

        # Obtener la imagen procesada desde base64
        img_data = data['processed_image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        
        # Guardar temporalmente la imagen procesada
        temp_input = os.path.join('temp', 'temp_processed.png')
        with open(temp_input, 'wb') as f:
            f.write(img_bytes)

        # Generar mockup
        output_path = mockup_generator.generate(temp_input, data['tipo'])
        
        # Convertir el mockup a base64
        with open(output_path, 'rb') as img_file:
            mockup_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Limpiar archivos temporales
        os.remove(temp_input)
        os.remove(output_path)
        
        return jsonify({
            'success': True,
            'mockup_image': f'data:image/png;base64,{mockup_data}'
        })
    except Exception as e:
        return jsonify({'error': str(e).encode('utf-8', errors='ignore').decode('utf-8')}), 500

@app.route('/processed/<path:filename>')
def processed_file(filename):
    try:
        return send_file(filename, as_attachment=False)
    except Exception as e:
        return jsonify({'error': str(e).encode('utf-8', errors='ignore').decode('utf-8')}), 404

@app.route('/analyze_size', methods=['POST'])
def analyze_logo_size():
    try:
        print("📥 Entrando a /analyze_size")
        print("Archivos recibidos:", request.files)
        print("Formulario recibido:", request.form)

        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        if 'target_size' not in request.form:
            return jsonify({'error': 'No target size specified'}), 400
        
        file = request.files['file']
        target_size = float(request.form['target_size'])
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)
            
            try:
                results = logo_size_analyzer.analyze_and_resize_logo(input_path, target_size)
                solicitado = results['resultados']['solicitado']
                output_filename = f'temp_solicitado_{filename}'
                output_path = os.path.join('temp', output_filename)
                cv2.imwrite(output_path, results['imagenes']['solicitado'])

                with open(output_path, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')

                os.remove(output_path)

                response_data = {
                    'resized_image': f'data:image/png;base64,{img_data}',
                    'respuesta': results['respuesta'],
                    'resultados': {
                        'original': {
                            'width': round(solicitado['width_mm']),
                            'height': round(solicitado['height_mm'])
                        },
                        'text_analysis': {
                            'texts_below_2mm': len([h for h in solicitado['text_heights_mm'] if h < 2.0]),
                            'min_text_height': round(min(solicitado['text_heights_mm']) if solicitado['text_heights_mm'] else 0),
                            'avg_text_height': round(sum(solicitado['text_heights_mm']) / len(solicitado['text_heights_mm']) if solicitado['text_heights_mm'] else 0)
                        },
                        'alternatives': {
                            'smaller': {
                                'size': round(target_size - 10),
                                'width': round(results['resultados']['chico']['width_mm']),
                                'height': round(results['resultados']['chico']['height_mm']),
                                'text_analysis': {
                                    'texts_below_2mm': len([h for h in results['resultados']['chico']['text_heights_mm'] if h < 2.0]),
                                    'min_text_height': round(min(results['resultados']['chico']['text_heights_mm']) if results['resultados']['chico']['text_heights_mm'] else 0),
                                    'avg_text_height': round(sum(results['resultados']['chico']['text_heights_mm']) / len(results['resultados']['chico']['text_heights_mm']) if results['resultados']['chico']['text_heights_mm'] else 0)
                                }
                            },
                            'larger': {
                                'size': round(target_size + 10),
                                'width': round(results['resultados']['grande']['width_mm']),
                                'height': round(results['resultados']['grande']['height_mm']),
                                'text_analysis': {
                                    'texts_below_2mm': len([h for h in results['resultados']['grande']['text_heights_mm'] if h < 2.0]),
                                    'min_text_height': round(min(results['resultados']['grande']['text_heights_mm']) if results['resultados']['grande']['text_heights_mm'] else 0),
                                    'avg_text_height': round(sum(results['resultados']['grande']['text_heights_mm']) / len(results['resultados']['grande']['text_heights_mm']) if results['resultados']['grande']['text_heights_mm'] else 0)
                                }
                            }
                        }
                    }
                }

                return jsonify(response_data)
            
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

            
        finally:
            # Limpiar el archivo de entrada
            if os.path.exists(input_path):
                os.remove(input_path)

if __name__ == '__main__':
    # Asegurar que existan los directorios necesarios
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('temp', exist_ok=True)
    os.makedirs('mockups', exist_ok=True)
    
    # Iniciar el servidor
    app.run(debug=True)
