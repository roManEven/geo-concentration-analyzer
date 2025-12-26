# app.py
from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
import pandas as pd
import folium
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import math
import os
import tempfile
import json
from werkzeug.utils import secure_filename
from datetime import datetime
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import time

app = Flask(__name__)
app.secret_key = 'temp_key_for_render'  # Фиксированный для Render
CORS(app)

# Настройки загрузки файлов
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB максимум для бесплатного плана
app.config['ALLOWED_EXTENSIONS'] = {'xlsx', 'xls', 'csv'}

# Лимиты для бесплатного Render
MAX_POINTS = 500  # Максимум точек для анализа
TIMEOUT_SECONDS = 25  # Таймаут операции

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def parse_coordinates_optimized(coord_str):
    """Быстрый парсинг координат"""
    if pd.isna(coord_str):
        return None
    
    try:
        # Упрощенный парсинг
        coord_str = str(coord_str).strip()
        if not coord_str:
            return None
            
        # Пробуем найти два числа
        import re
        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', coord_str)
        
        if len(numbers) >= 2:
            return [float(numbers[0]), float(numbers[1])]
    except:
        pass
    return None

def find_concentrations_optimized(coords, radius_m=500, min_points=10, max_results=10):
    """Оптимизированный поиск концентраций с DBSCAN"""
    if len(coords) <= 1:
        return []
    
    # Преобразуем в numpy для скорости
    coords_array = np.array(coords)
    
    # Используем DBSCAN для кластеризации
    # Переводим радиус в градусы (приблизительно)
    eps = radius_m / 111000  # 1 градус ≈ 111км
    
    try:
        dbscan = DBSCAN(eps=eps, min_samples=min_points, metric='euclidean', n_jobs=1)
        labels = dbscan.fit_predict(coords_array)
    except:
        # Fallback на простой алгоритм для малых наборов
        return find_concentrations_simple(coords, radius_m, min_points, max_results)
    
    concentrations = []
    unique_labels = set(labels)
    
    for label in unique_labels:
        if label == -1:  # Шум
            continue
            
        mask = labels == label
        cluster_points = coords_array[mask]
        
        if len(cluster_points) >= min_points:
            # Центр кластера
            center = cluster_points.mean(axis=0)
            
            # Рассчитываем среднее расстояние
            distances = []
            for point in cluster_points:
                try:
                    dist = geodesic(center, point).meters
                    distances.append(dist)
                except:
                    distances.append(0)
            
            avg_distance = np.mean(distances) if distances else 0
            
            # Получаем индексы точек
            point_indices = np.where(mask)[0].tolist()
            
            concentrations.append({
                'center': center.tolist(),
                'count': len(cluster_points),
                'points': point_indices,
                'avg_distance': avg_distance,
                'density': len(cluster_points) / (math.pi * (radius_m/1000) ** 2) if radius_m > 0 else 0
            })
    
    # Сортируем и ограничиваем результаты
    concentrations.sort(key=lambda x: x['count'], reverse=True)
    
    # Получаем адреса только для топ-3 (для экономии времени)
    geolocator = None
    for i, conc in enumerate(concentrations[:3]):
        try:
            if geolocator is None:
                geolocator = Nominatim(user_agent="concentration_analyzer", timeout=5)
            location = geolocator.reverse(f"{conc['center'][0]}, {conc['center'][1]}", language='ru')
            conc['address'] = location.address if location else f"Координаты: {conc['center'][0]:.6f}, {conc['center'][1]:.6f}"
        except:
            conc['address'] = f"Координаты: {conc['center'][0]:.6f}, {conc['center'][1]:.6f}"
    
    for conc in concentrations[3:]:
        conc['address'] = f"Координаты: {conc['center'][0]:.6f}, {conc['center'][1]:.6f}"
    
    return concentrations[:max_results]

def find_concentrations_simple(coords, radius_m=500, min_points=10, max_results=10):
    """Простой алгоритм для малых наборов данных"""
    concentrations = []
    processed = set()
    
    # Ограничиваем количество проверок
    max_checks = min(100, len(coords))
    
    for i in range(max_checks):
        if i in processed:
            continue
            
        center = coords[i]
        neighbors = [i]
        
        for j in range(len(coords)):
            if j == i or j in processed:
                continue
                
            try:
                if geodesic(center, coords[j]).meters <= radius_m:
                    neighbors.append(j)
            except:
                continue
        
        if len(neighbors) >= min_points:
            # Вычисляем центр масс
            neighbor_coords = [coords[idx] for idx in neighbors]
            center_lat = sum(c[0] for c in neighbor_coords) / len(neighbor_coords)
            center_lon = sum(c[1] for c in neighbor_coords) / len(neighbor_coords)
            
            concentrations.append({
                'center': [center_lat, center_lon],
                'count': len(neighbors),
                'points': neighbors,
                'avg_distance': radius_m / 2,  # Примерное значение
                'density': len(neighbors) / (math.pi * (radius_m/1000) ** 2)
            })
            
            processed.update(neighbors)
    
    # Получаем адреса
    for conc in concentrations[:5]:
        try:
            geolocator = Nominatim(user_agent="concentration_analyzer", timeout=5)
            location = geolocator.reverse(f"{conc['center'][0]}, {conc['center'][1]}", language='ru')
            conc['address'] = location.address if location else f"Координаты: {conc['center'][0]:.6f}, {conc['center'][1]:.6f}"
        except:
            conc['address'] = f"Координаты: {conc['center'][0]:.6f}, {conc['center'][1]:.6f}"
    
    return concentrations[:max_results]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preview_columns', methods=['POST'])
def preview_columns():
    """Предпросмотр столбцов - оптимизированная версия"""
    start_time = time.time()
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Читаем только первые строки для предпросмотра
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file, nrows=5)
        else:
            df = pd.read_excel(file, nrows=5)
        
        # Проверяем таймаут
        if time.time() - start_time > TIMEOUT_SECONDS:
            return jsonify({'error': 'Timeout reading file'}), 408
        
        columns = df.columns.tolist()
        sample_data = df.head(3).to_dict(orient='records')
        
        return jsonify({
            'success': True,
            'columns': columns,
            'sample_data': sample_data,
            'row_count': len(df)
        })
        
    except Exception as e:
        print(f"Preview error: {str(e)}")
        return jsonify({'error': f'Error reading file: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Основной анализ с оптимизациями"""
    start_time = time.time()
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Получаем параметры
        data = request.form
        column_name = data.get('column_name', '')
        radius_m = min(float(data.get('radius_m', 500)), 2000)  # Ограничиваем радиус
        min_points = min(int(data.get('min_points', 5)), 50)    # Ограничиваем min_points
        max_results = min(int(data.get('max_results', 5)), 10)  # Ограничиваем результаты
        
        # Читаем файл с ограничением строк
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file, nrows=MAX_POINTS * 2)  # Читаем больше для фильтрации
        else:
            df = pd.read_excel(file, nrows=MAX_POINTS * 2)
        
        if time.time() - start_time > TIMEOUT_SECONDS:
            return jsonify({'error': 'File reading timeout'}), 408
        
        # Парсим координаты
        coordinates = []
        for idx, row in df.iterrows():
            if len(coordinates) >= MAX_POINTS:
                break
                
            coord = parse_coordinates_optimized(row.get(column_name, ''))
            if coord:
                coordinates.append(coord)
        
        if len(coordinates) < min_points:
            return jsonify({
                'error': f'Необходимо минимум {min_points} точек для анализа. Найдено: {len(coordinates)}'
            }), 400
        
        # Находим концентрации
        concentrations = find_concentrations_optimized(
            coordinates, 
            radius_m=radius_m, 
            min_points=min_points,
            max_results=max_results
        )
        
        if time.time() - start_time > TIMEOUT_SECONDS:
            return jsonify({'error': 'Analysis timeout'}), 408
        
        # Создаем упрощенную карту
        if coordinates:
            center_lat = sum(c[0] for c in coordinates) / len(coordinates)
            center_lon = sum(c[1] for c in coordinates) / len(coordinates)
        else:
            center_lat, center_lon = 55.7558, 37.6173
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='OpenStreetMap',
            control_scale=True
        )
        
        # Добавляем точки (только первые 100 для скорости)
        for i, (lat, lon) in enumerate(coordinates[:100]):
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color='blue',
                fill=True,
                popup=f'Точка {i+1}: {lat:.4f}, {lon:.4f}'
            ).add_to(m)
        
        # Добавляем концентрации
        colors = ['red', 'orange', 'green', 'purple', 'darkred']
        for i, conc in enumerate(concentrations):
            color = colors[i % len(colors)]
            
            folium.Circle(
                location=conc['center'],
                radius=radius_m,
                color=color,
                fill=True,
                fill_opacity=0.2,
                popup=f'''Концентрация #{i+1}<br>
                         Точек: {conc['count']}<br>
                         Адрес: {conc.get('address', 'Нет данных')}'''
            ).add_to(m)
            
            folium.Marker(
                location=conc['center'],
                popup=f'Центр #{i+1}',
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(m)
        
        # Сохраняем карту
        map_filename = f"map_{datetime.now().strftime('%H%M%S')}.html"
        map_path = os.path.join(app.config['UPLOAD_FOLDER'], map_filename)
        m.save(map_path)
        
        # Очищаем старые файлы
        cleanup_old_files()
        
        return jsonify({
            'success': True,
            'total_points': len(coordinates),
            'concentrations_found': len(concentrations),
            'concentrations': concentrations,
            'map_url': f'/get_map/{map_filename}',
            'processing_time': round(time.time() - start_time, 2),
            'message': f'Проанализировано {len(coordinates)} точек. Найдено {len(concentrations)} концентраций.'
        })
        
    except Exception as e:
        print(f"Analyze error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def cleanup_old_files():
    """Очистка старых файлов"""
    try:
        temp_dir = app.config['UPLOAD_FOLDER']
        current_time = time.time()
        
        for filename in os.listdir(temp_dir):
            if filename.startswith(('map_', 'results_')):
                filepath = os.path.join(temp_dir, filename)
                try:
                    # Удаляем файлы старше 30 минут
                    if current_time - os.path.getmtime(filepath) > 1800:
                        os.remove(filepath)
                except:
                    pass
    except:
        pass

@app.route('/get_map/<filename>')
def get_map(filename):
    """Отдача файла карты"""
    try:
        map_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(map_path):
            return send_file(map_path, mimetype='text/html')
        return 'Карта не найдена', 404
    except Exception as e:
        return str(e), 500

@app.route('/health')
def health():
    """Health check для Render"""
    return jsonify({
        'status': 'ok',
        'service': 'geo-concentration-analyzer',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
