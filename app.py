# app.py
from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
import pandas as pd
import folium
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import math
import os
import tempfile
import json
from werkzeug.utils import secure_filename
from datetime import datetime
import numpy as np
import shutil

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Случайный секретный ключ
CORS(app)

# Настройки загрузки файлов
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['ALLOWED_EXTENSIONS'] = {'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def parse_coordinates(coord_str):
    """Парсинг координат из строки"""
    try:
        coord_str = str(coord_str).strip()
        
        if pd.isna(coord_str):
            return None
            
        # Убираем возможные скобки и кавычки
        coord_str = coord_str.replace('(', '').replace(')', '')
        coord_str = coord_str.replace('[', '').replace(']', '')
        coord_str = coord_str.replace('"', '').replace("'", '')
        
        if ',' in coord_str:
            parts = coord_str.split(',')
            if len(parts) == 2:
                return [float(parts[0].strip()), float(parts[1].strip())]
            elif len(parts) > 2:
                # Может быть формат "lat, lon, something"
                return [float(parts[0].strip()), float(parts[1].strip())]
        
        elif ' ' in coord_str:
            parts = coord_str.split()
            if len(parts) >= 2:
                # Пробуем найти два числа
                nums = []
                for part in parts:
                    try:
                        num = float(part)
                        nums.append(num)
                        if len(nums) == 2:
                            return nums
                    except:
                        continue
        
        elif ';' in coord_str:
            parts = coord_str.split(';')
            if len(parts) == 2:
                return [float(parts[0].strip()), float(parts[1].strip())]
                
    except Exception as e:
        print(f"Ошибка парсинга '{coord_str}': {str(e)}")
        return None
    return None

def get_address(lat, lon):
    """Получение адреса по координатам"""
    try:
        geolocator = Nominatim(user_agent="concentration_analyzer", timeout=10)
        location = geolocator.reverse(f"{lat}, {lon}", language='ru')
        if location and location.address:
            return location.address
    except Exception as e:
        print(f"Ошибка геокодирования: {str(e)}")
    return f"Координаты: {lat:.6f}, {lon:.6f}"

def create_circle(center, radius_km, num_points=100):
    """Создает полигон круга"""
    circle_points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        dx = radius_km * 1000 * math.cos(angle) / 111320
        dy = radius_km * 1000 * math.sin(angle) / (111320 * math.cos(math.radians(center[0])))
        
        lat = center[0] + dx
        lon = center[1] + dy
        circle_points.append([lat, lon])
    
    circle_points.append(circle_points[0])
    return circle_points

def find_concentrations(coords, radius_m=500, min_points=10, max_results=10):
    """Поиск мест с высокой концентрацией точек"""
    concentrations = []
    processed = set()
    
    for i, center in enumerate(coords):
        if i in processed:
            continue
            
        neighbors = []
        distances = []
        
        for j, point in enumerate(coords):
            if j == i:
                continue
                
            try:
                dist = geodesic(center, point).meters
                if dist <= radius_m:
                    neighbors.append(j)
                    distances.append(dist)
            except Exception as e:
                print(f"Ошибка расчета расстояния: {str(e)}")
                continue
        
        total_points = len(neighbors) + 1
        
        if total_points >= min_points:
            # Вычисляем центр масс
            neighbor_coords = [coords[idx] for idx in neighbors] + [center]
            center_lat = sum(c[0] for c in neighbor_coords) / total_points
            center_lon = sum(c[1] for c in neighbor_coords) / total_points
            
            concentrations.append({
                'center': [center_lat, center_lon],
                'count': total_points,
                'points': neighbors + [i],
                'distances': distances + [0]
            })
            
            processed.update(neighbors)
            processed.add(i)
    
    # Сортируем по количеству точек
    concentrations.sort(key=lambda x: x['count'], reverse=True)
    
    # Получаем адреса для топ результатов
    for conc in concentrations[:max_results]:
        conc['address'] = get_address(conc['center'][0], conc['center'][1])
        if conc['distances']:
            conc['avg_distance'] = sum(conc['distances']) / len(conc['distances'])
        else:
            conc['avg_distance'] = 0
            
        area_km2 = math.pi * (radius_m / 1000) ** 2
        conc['density'] = conc['count'] / area_km2 if area_km2 > 0 else 0
    
    return concentrations[:max_results]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preview_columns', methods=['POST'])
def preview_columns():
    """Предпросмотр столбцов Excel файла"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Сохраняем временно
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(temp_path)
        
        # Читаем только заголовки
        df = pd.read_excel(temp_path, nrows=1)
        columns = df.columns.tolist()
        
        # Очищаем
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'columns': columns,
            'row_count': len(df)
        })
        
    except Exception as e:
        print(f"Error in preview_columns: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Основной анализ данных"""
    try:
        # Получаем данные из формы
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Получаем параметры
        data = request.form
        column_name = data.get('column_name', '')
        radius_m = float(data.get('radius_m', 500))
        min_points = int(data.get('min_points', 10))
        max_results = int(data.get('max_results', 10))
        
        if not column_name:
            return jsonify({'error': 'Не выбран столбец с координатами'}), 400
        
        # Сохраняем файл
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Читаем Excel
        try:
            df = pd.read_excel(filepath)
        except Exception as e:
            os.remove(filepath)
            return jsonify({'error': f'Ошибка чтения Excel файла: {str(e)}'}), 400
        
        # Проверяем существование столбца
        if column_name not in df.columns:
            os.remove(filepath)
            return jsonify({'error': f'Столбец "{column_name}" не найден в файле'}), 400
        
        # Парсим координаты
        coordinates = []
        valid_rows = []
        
        for idx, row in df.iterrows():
            coord = parse_coordinates(row[column_name])
            if coord:
                coordinates.append(coord)
                valid_rows.append(idx + 1)
        
        if len(coordinates) < 2:
            os.remove(filepath)
            return jsonify({'error': 'Необходимо минимум 2 точки для анализа. Проверьте формат координат.'}), 400
        
        # Находим концентрации
        concentrations = find_concentrations(
            coordinates, 
            radius_m=radius_m, 
            min_points=min_points,
            max_results=max_results
        )
        
        # Создаем карту
        if coordinates:
            center_lat = sum(c[0] for c in coordinates) / len(coordinates)
            center_lon = sum(c[1] for c in coordinates) / len(coordinates)
        else:
            center_lat, center_lon = 55.7558, 37.6173  # Москва по умолчанию
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            control_scale=True,
            tiles='OpenStreetMap'
        )
        
        # Добавляем все точки
        for i, (lat, lon) in enumerate(coordinates, 1):
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.7,
                popup=f'<b>Точка {i}</b><br>Широта: {lat:.6f}<br>Долгота: {lon:.6f}'
            ).add_to(m)
        
        # Добавляем концентрации
        colors = ['red', 'orange', 'green', 'purple', 'darkred', 'darkblue']
        
        for i, conc in enumerate(concentrations):
            color = colors[i % len(colors)]
            
            # Круг концентрации
            conc_radius_km = radius_m / 1000
            circle_points = create_circle(conc['center'], conc_radius_km)
            
            folium.Polygon(
                locations=circle_points,
                color=color,
                weight=3,
                fill=True,
                fill_color=color,
                fill_opacity=0.2,
                popup=f'<b>Концентрация #{i+1}</b><br>{conc["address"]}<br>Точек: {conc["count"]}<br>Ср. расстояние: {conc["avg_distance"]:.1f} м'
            ).add_to(m)
            
            # Центр концентрации
            folium.Marker(
                location=conc['center'],
                popup=f'<b>Центр концентрации #{i+1}</b><br>{conc["address"]}<br>Точек: {conc["count"]}<br>Плотность: {conc["density"]:.1f} точек/км²',
                icon=folium.Icon(color=color, icon='star', prefix='fa')
            ).add_to(m)
            
            # Точки в концентрации
            for point_idx in conc['points']:
                if point_idx < len(coordinates):
                    lat, lon = coordinates[point_idx]
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=6,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.9,
                        popup=f'Точка {point_idx + 1} в концентрации #{i+1}'
                    ).add_to(m)
        
        # Находим максимальное расстояние
        max_distance = 0
        point1_idx = point2_idx = 0
        
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                try:
                    dist = geodesic(coordinates[i], coordinates[j]).meters
                    if dist > max_distance:
                        max_distance = dist
                        point1_idx = i
                        point2_idx = j
                except:
                    continue
        
        if max_distance > 0:
            folium.PolyLine(
                locations=[coordinates[point1_idx], coordinates[point2_idx]],
                color='black',
                weight=3,
                opacity=0.7,
                dash_array='10, 10',
                popup=f'Диаметр: {max_distance / 1000:.2f} км'
            ).add_to(m)
        
        # Добавляем тепловую карту
        heat_data = [[coord[0], coord[1], 1] for coord in coordinates]
        from folium.plugins import HeatMap
        HeatMap(heat_data, radius=20, blur=15, max_zoom=1).add_to(m)
        
        # Добавляем мини-карту
        from folium.plugins import MiniMap
        minimap = MiniMap(toggle_display=True)
        m.add_child(minimap)
        
        # Добавляем полноэкранный режим
        from folium.plugins import Fullscreen
        Fullscreen().add_to(m)
        
        # Легенда
        legend_html = f'''
        <div style="position: fixed; bottom: 50px; left: 50px; width: 300px; 
             background-color: white; border:2px solid grey; z-index:9999; 
             font-size:14px; padding: 10px; border-radius: 5px; opacity: 0.9;">
             <b>Анализ концентраций</b><hr style="margin:5px 0">
             <b>Всего точек:</b> {len(coordinates)}<br>
             <b>Концентраций:</b> {len(concentrations)}<br>
             <b>Диаметр:</b> {max_distance/1000:.2f} км<hr style="margin:5px 0">
             <span style="color:blue">●</span> Все точки<br>
             <span style="color:red">★</span> Центры концентраций<br>
             <span style="color:black">━━━━━</span> Диаметр
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Сохраняем данные в сессию
        session['coordinates'] = json.dumps(coordinates)
        session['concentrations'] = json.dumps(concentrations)
        session['analysis_params'] = {
            'radius_m': radius_m,
            'min_points': min_points,
            'total_points': len(coordinates)
        }
        
        # Сохраняем карту
        map_filename = f"map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        map_path = os.path.join(app.config['UPLOAD_FOLDER'], map_filename)
        m.save(map_path)
        
        session['map_filename'] = map_filename
        
        # Очищаем временный файл Excel
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'total_points': len(coordinates),
            'concentrations_found': len(concentrations),
            'concentrations': concentrations,
            'max_distance_km': max_distance / 1000,
            'map_url': f'/get_map/{map_filename}',
            'message': f'Проанализировано {len(coordinates)} точек. Найдено {len(concentrations)} концентраций.'
        })
        
    except Exception as e:
        print(f"Error in analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_map/<filename>')
def get_map(filename):
    """Отдача файла карты"""
    try:
        map_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(map_path):
            return send_file(map_path)
        return 'Карта не найдена', 404
    except Exception as e:
        print(f"Error in get_map: {str(e)}")
        return str(e), 500

@app.route('/export_excel', methods=['POST'])
def export_excel():
    """Экспорт результатов в Excel"""
    try:
        data = request.json
        concentrations = data.get('concentrations', [])
        
        if not concentrations:
            return jsonify({'error': 'Нет данных для экспорта'}), 400
        
        # Создаем DataFrame
        df_data = []
        for i, conc in enumerate(concentrations, 1):
            df_data.append({
                '№': i,
                'Адрес': conc.get('address', ''),
                'Широта центра': conc['center'][0],
                'Долгота центра': conc['center'][1],
                'Количество точек': conc['count'],
                'Среднее расстояние (м)': round(conc.get('avg_distance', 0), 1),
                'Плотность (точек/км²)': round(conc.get('density', 0), 1),
                'Индексы точек': ', '.join(str(p + 1) for p in conc.get('points', []))
            })
        
        df = pd.DataFrame(df_data)
        
        # Сохраняем во временный файл
        export_filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        export_path = os.path.join(app.config['UPLOAD_FOLDER'], export_filename)
        df.to_excel(export_path, index=False)
        
        return jsonify({
            'success': True,
            'download_url': f'/download/{export_filename}',
            'filename': export_filename
        })
        
    except Exception as e:
        print(f"Error in export_excel: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Скачивание файла"""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            response = send_file(file_path, as_attachment=True)
            
            # Очищаем файл после отправки (в фоновом режиме)
            def cleanup_file():
                try:
                    os.remove(file_path)
                except:
                    pass
            
            # Запускаем очистку после ответа
            from threading import Timer
            Timer(5.0, cleanup_file).start()
            
            return response
        return 'Файл не найден', 404
    except Exception as e:
        print(f"Error in download_file: {str(e)}")
        return str(e), 500

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Очистка временных файлов"""
    try:
        # Удаляем старые файлы карт и результатов
        temp_dir = app.config['UPLOAD_FOLDER']
        current_time = datetime.now()
        
        for filename in os.listdir(temp_dir):
            if filename.startswith(('map_', 'results_')):
                filepath = os.path.join(temp_dir, filename)
                try:
                    # Удаляем файлы старше 1 часа
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    if (current_time - file_time).total_seconds() > 3600:
                        os.remove(filepath)
                except:
                    pass
        
        return jsonify({
            'success': True,
            'message': 'Временные файлы очищены'
        })
    except Exception as e:
        print(f"Error in cleanup: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Проверка работоспособности"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'Файл слишком большой. Максимальный размер: 16MB'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Страница не найдена'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

if __name__ == '__main__':
    # Создаем временную папку если её нет
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Запускаем Flask
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
