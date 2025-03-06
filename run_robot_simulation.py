import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Conv3D, Flatten, Input, Concatenate, LSTM, TimeDistributed
from tensorflow.keras.layers import BatchNormalization, Dropout, MaxPooling2D, AveragePooling2D, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter
import cv2
import gym
import pygame
import threading
import queue
import os
import json
import pickle
import sys
from tqdm import tqdm

# ANSI-Farbcodes für hübschere Konsolenausgabe
class TermColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# GPU-Erkennung und erweiterte Konfiguration
class HardwareManager:
    """
    Erkennt und optimiert die verfügbare Hardware für die Simulation.
    """
    def __init__(self):
        self.gpu_info = self._detect_gpus()
        self.cpu_info = self._detect_cpu()
        self.ram_info = self._detect_ram()
        self.performance_profile = self._calculate_performance_profile()
        
        print(f"{TermColors.HEADER}Hardware-Erkennung:{TermColors.ENDC}")
        print(f"  {TermColors.BOLD}GPUs:{TermColors.ENDC} {len(self.gpu_info['devices'])} gefunden")
        for i, gpu in enumerate(self.gpu_info['devices']):
            print(f"    - GPU {i+1}: {gpu['name']} ({gpu['memory_mb']}MB)")
        print(f"  {TermColors.BOLD}CPU:{TermColors.ENDC} {self.cpu_info['cores']} Kerne @ {self.cpu_info['frequency_ghz']:.2f} GHz")
        print(f"  {TermColors.BOLD}RAM:{TermColors.ENDC} {self.ram_info['total_gb']:.1f} GB")
        print(f"  {TermColors.BOLD}Performance-Profil:{TermColors.ENDC} {self.performance_profile['level']}")
        
        # TensorFlow für optimale Leistung konfigurieren
        self._configure_tensorflow()
    
    def _detect_gpus(self):
        """
        Erkennt verfügbare GPUs und ihre Eigenschaften.
        """
        gpus = tf.config.list_physical_devices('GPU')
        gpu_info = {
            'available': len(gpus) > 0,
            'devices': []
        }
        
        if gpus:
            for gpu in gpus:
                try:
                    # GPU-Speicher nach Bedarf zuweisen
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # GPU-Name und andere Details abrufen
                    try:
                        gpu_details = tf.config.experimental.get_device_details(gpu)
                    except:
                        gpu_details = {'device_name': 'GPU', 'compute_capability': 0.0}
                    
                    # Benchmark durchführen
                    with tf.device(gpu.name):
                        # Matrix-Multiplikation als Benchmark
                        start_time = time.time()
                        a = tf.random.normal([3000, 3000])
                        b = tf.random.normal([3000, 3000])
                        c = tf.matmul(a, b)
                        tf.reduce_sum(c)  # Berechnung erzwingen
                        elapsed = time.time() - start_time
                        
                        # Performance-Score berechnen
                        perf_score = 5.0 / max(0.1, elapsed)  # Höherer Wert = bessere Leistung
                        
                        # Speichergröße abschätzen (in MB)
                        if 'compute_capability' in gpu_details:
                            # Versuch, die Speichergröße anhand der Compute Capability zu schätzen
                            cc = gpu_details['compute_capability']
                            if cc >= 7.0:  # Neuere GPUs haben tendenziell mehr Speicher
                                memory_mb = 8192
                            elif cc >= 6.0:
                                memory_mb = 4096
                            else:
                                memory_mb = 2048
                        else:
                            # Fallback-Wert
                            memory_mb = 4096
                
                    gpu_info['devices'].append({
                        'name': gpu_details.get('device_name', 'Unknown GPU'),
                        'compute_capability': gpu_details.get('compute_capability', 0.0),
                        'performance_score': perf_score,
                        'memory_mb': memory_mb
                    })
                
                except Exception as e:
                    print(f"GPU-Konfigurationsfehler: {e}")
                    # Füge ein Dummy-Device hinzu, damit die Logik weiterhin funktioniert
                    gpu_info['devices'].append({
                        'name': 'Unknown GPU',
                        'compute_capability': 0.0,
                        'performance_score': 1.0,
                        'memory_mb': 2048
                    })
        
        return gpu_info
    
    def _detect_cpu(self):
        """
        Erkennt CPU-Eigenschaften wie Anzahl der Kerne.
        """
        import multiprocessing
        import platform
        
        cores = multiprocessing.cpu_count()
        
        # Versuchen, die CPU-Taktfrequenz zu bestimmen
        freq_ghz = 0.0
        try:
            if platform.system() == 'Linux':
                # Unter Linux kann die maximale Frequenz über das Dateisystem abgefragt werden
                try:
                    with open('/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq', 'r') as f:
                        freq_khz = int(f.read().strip())
                        freq_ghz = freq_khz / 1000000.0
                except:
                    freq_ghz = 2.5
            else:
                # Fallback-Wert
                freq_ghz = 2.5
        except:
            # Wenn die Abfrage fehlschlägt, einen Standardwert verwenden
            freq_ghz = 2.5
        
        # Einfacher CPU-Benchmark
        start_time = time.time()
        np.random.seed(0)
        a = np.random.rand(1500, 1500)
        b = np.random.rand(1500, 1500)
        c = np.dot(a, b)
        elapsed = time.time() - start_time
        
        # Performance-Score
        perf_score = 3.0 / max(0.1, elapsed)
        
        return {
            'cores': cores,
            'frequency_ghz': freq_ghz,
            'performance_score': perf_score
        }
    
    def _detect_ram(self):
        """
        Erkennt verfügbaren Arbeitsspeicher.
        """
        try:
            import psutil
            
            # Gesamter und verfügbarer Arbeitsspeicher in GB
            total_ram = psutil.virtual_memory().total / (1024**3)
            available_ram = psutil.virtual_memory().available / (1024**3)
            
            return {
                'total_gb': total_ram,
                'available_gb': available_ram
            }
        except:
            # Fallback-Werte, wenn psutil nicht verfügbar ist
            return {
                'total_gb': 8.0,
                'available_gb': 4.0
            }
    
    def _calculate_performance_profile(self):
        """
        Berechnet ein Gesamtleistungsprofil basierend auf CPU, GPU und RAM.
        """
        # Grundwert basierend auf CPU
        cpu_score = self.cpu_info['performance_score'] * min(self.cpu_info['cores'], 16) / 8
        
        # GPU-Beitrag
        gpu_score = 0
        if self.gpu_info['available'] and len(self.gpu_info['devices']) > 0:
            for gpu in self.gpu_info['devices']:
                gpu_score += gpu['performance_score'] * (gpu['memory_mb'] / 4096)
        
        # RAM-Beitrag
        ram_score = min(2.0, self.ram_info.get('available_gb', 4.0) / 8.0)
        
        # Gesamtbewertung
        total_score = cpu_score * 0.3 + gpu_score * 0.6 + ram_score * 0.1
        
        # Klassifizieren
        if total_score > 15:
            level = "Ultra"
            max_resolution = 128
            max_entities = 500
            physics_detail = 4  # 4 = höchstes Detail
        elif total_score > 10:
            level = "High"
            max_resolution = 96
            max_entities = 300
            physics_detail = 3
        elif total_score > 5:
            level = "Medium"
            max_resolution = 64
            max_entities = 150
            physics_detail = 2
        else:
            level = "Low"
            max_resolution = 48
            max_entities = 80
            physics_detail = 1
        
        return {
            'score': total_score,
            'level': level,
            'max_resolution': max_resolution,
            'max_entities': max_entities,
            'physics_detail': physics_detail
        }
    
    def _configure_tensorflow(self):
        """
        Konfiguriert TensorFlow für optimale Leistung.
        """
        try:
            # Parallelisierungseinstellungen
            tf.config.threading.set_inter_op_parallelism_threads(
                min(self.cpu_info['cores'], 8))  # Begrenzen auf 8 Threads
            tf.config.threading.set_intra_op_parallelism_threads(
                min(self.cpu_info['cores'], 8))
            
            # Mixed Precision Training aktivieren, wenn verfügbar
            if self.gpu_info['available'] and len(self.gpu_info['devices']) > 0:
                try:
                    # TensorFlow Version überprüfen
                    if hasattr(tf.keras.mixed_precision, 'set_global_policy'):
                        policy = tf.keras.mixed_precision.Policy('mixed_float16')
                        tf.keras.mixed_precision.set_global_policy(policy)
                        print(f"  {TermColors.OKGREEN}Mixed Precision Training aktiviert{TermColors.ENDC}")
                    else:
                        print(f"  {TermColors.WARNING}Mixed Precision Training nicht verfügbar (TF-Version < 2.4){TermColors.ENDC}")
                except Exception as e:
                    print(f"  {TermColors.WARNING}Mixed Precision Training nicht verfügbar: {e}{TermColors.ENDC}")
        except Exception as e:
            print(f"  {TermColors.WARNING}TensorFlow-Konfiguration fehlgeschlagen: {e}{TermColors.ENDC}")
    
    def get_simulation_config(self):
        """
        Liefert optimale Konfigurationsparameter für die Simulation basierend auf der Hardware.
        """
        p = self.performance_profile
        
        # Adaptive Parameter für die Simulation
        config = {
            'resolution': p['max_resolution'],
            'max_entities': p['max_entities'],
            'physics_detail': p['physics_detail'],
            'sensor_noise_levels': 0.5 if p['level'] in ['Low', 'Medium'] else 0.2,
            'render_quality': p['level'],
            'num_environments': max(1, min(2, len(self.gpu_info['devices']))),
            'batch_size': 16 if p['level'] == 'Low' else 32 if p['level'] == 'Medium' else 64 if p['level'] == 'High' else 128,
            'memory_limit': int(0.7 * self.ram_info.get('available_gb', 4.0) * 1024),  # 70% des verfügbaren RAMs in MB
        }
        
        return config


# Physik-Engine für realistische Simulation
class PhysicsEngine:
    """
    Physik-Engine für realistische Simulation von Roboter und Umgebung.
    """
    def __init__(self, detail_level=2, time_step=0.01):
        self.detail_level = detail_level  # 1 (niedrig) bis 4 (ultra)
        self.time_step = time_step
        self.gravity = np.array([0, 0, -9.81])
        self.collision_tolerance = 0.001
        
        # Physikalische Konstanten und Parameter
        self.friction_coefficients = {
            'concrete': 0.7,
            'metal': 0.2,
            'grass': 0.35,
            'sand': 0.6,
            'ice': 0.05,
            'rubber': 0.9
        }
        
        # Kollisionserkennung und -auflösung aktivieren basierend auf Detail-Level
        self.collision_iterations = {
            1: 1,   # Niedrig: Einfache Kollisionserkennung
            2: 3,   # Mittel: Mehrere Iterationen
            3: 5,   # Hoch: Viele Iterationen für präzisere Physik
            4: 8    # Ultra: Sehr präzise Kollisionsauflösung
        }[detail_level]
        
        # Dynamik-Parameter
        self.elasticity = 0.3  # Elastizität bei Kollisionen
        
        # Eigenschaften von Materialien
        self.materials = {
            'concrete': {'density': 2400, 'elasticity': 0.2, 'roughness': 0.7},
            'metal': {'density': 7800, 'elasticity': 0.15, 'roughness': 0.2},
            'grass': {'density': 300, 'elasticity': 0.4, 'roughness': 0.8},
            'sand': {'density': 1600, 'elasticity': 0.1, 'roughness': 0.9},
            'ice': {'density': 900, 'elasticity': 0.01, 'roughness': 0.05},
            'rubber': {'density': 1100, 'elasticity': 0.7, 'roughness': 0.9},
            'wood': {'density': 700, 'elasticity': 0.3, 'roughness': 0.6}
        }
        
    def apply_forces(self, entity, forces, torques, dt=None):
        """
        Wendet Kräfte und Drehmomente auf eine Entität an und aktualisiert Position und Rotation.
        """
        if dt is None:
            dt = self.time_step
        
        # Masse und Trägheitsmoment aus der Entität abrufen
        mass = entity.get('mass', 1.0)
        inertia = entity.get('inertia', np.array([1.0, 1.0, 1.0]))
        
        # Aktuelle Geschwindigkeit und Winkelgeschwindigkeit
        velocity = entity.get('velocity', np.zeros(3))
        angular_velocity = entity.get('angular_velocity', np.zeros(3))
        
        # Beschleunigung berechnen (F=ma)
        acceleration = forces / mass
        if self.detail_level >= 2:
            acceleration += self.gravity  # Gravitation hinzufügen
        
        # Winkelgeschwindigkeit aktualisieren (τ=Iα)
        angular_acceleration = torques / inertia
        angular_velocity += angular_acceleration * dt
        
        # Geschwindigkeit aktualisieren
        velocity += acceleration * dt
        
        # Luftwiderstand simulieren (proportional zum Quadrat der Geschwindigkeit)
        if self.detail_level >= 3:
            # Konstante für Luftwiderstand
            drag_constant = 0.5 * 1.225 * entity.get('drag_coefficient', 0.5) * entity.get('cross_section', 1.0)
            velocity_squared = velocity * np.abs(velocity)
            drag_force = -drag_constant * velocity_squared
            velocity += (drag_force / mass) * dt
        
        # Position aktualisieren
        position = entity.get('position', np.zeros(3))
        position += velocity * dt
        
        # Rotation aktualisieren
        if self.detail_level >= 2:
            # Vereinfachte Quaternionen-Update für die Rotation
            current_rotation = entity.get('rotation', np.zeros(3))
            current_rotation += angular_velocity * dt
            
            # In den Bereich [-π, π] normalisieren
            current_rotation = np.mod(current_rotation + np.pi, 2 * np.pi) - np.pi
            
            # Aktualisiere die Rotation
            entity['rotation'] = current_rotation
            
            # Aktualisiere auch das Quaternion, falls vorhanden
            if 'rotation_quat' in entity:
                try:
                    # Berechne Quaternion aus Euler-Winkeln
                    roll, pitch, yaw = current_rotation
                    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
                    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
                    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
                    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
                
                    entity['rotation_quat'] = np.array([qw, qx, qy, qz])
                except:
                    # Fallback bei numerischen Problemen
                    entity['rotation_quat'] = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            # Einfachere Rotation für niedrigeres Detail-Level
            rotation = entity.get('rotation', np.zeros(3))
            rotation += angular_velocity * dt
            entity['rotation'] = rotation
        
        # Aktualisiere die Entität
        entity['position'] = position
        entity['velocity'] = velocity
        entity['angular_velocity'] = angular_velocity
        
        return entity
    
    def _check_entity_collision(self, entity1, entity2):
        """
        Überprüft Kollision zwischen zwei Entitäten.
        Gibt Kollisionsinformationen zurück, wenn eine Kollision erkannt wird.
        """
        # Positionen und Kollisionsradien abrufen
        pos1 = entity1.get('position', np.zeros(3))
        pos2 = entity2.get('position', np.zeros(3))
        radius1 = entity1.get('collision_radius', 0.5)
        radius2 = entity2.get('collision_radius', 0.5)
        
        # Abstand zwischen den Entitäten berechnen
        distance_vector = pos2 - pos1
        distance = np.linalg.norm(distance_vector)
        
        # Kollisionserkennung: Wenn der Abstand kleiner als die Summe der Radien ist
        if distance < radius1 + radius2:
            # Normalisiere den Richtungsvektor
            if distance > 0:
                normal = distance_vector / distance
            else:
                # Wenn die Entitäten an der gleichen Position sind, verwenden wir einen Standard-Normalvektor
                normal = np.array([1.0, 0.0, 0.0])
            
            # Überlappung berechnen
            overlap = (radius1 + radius2) - distance
            
            # Kollisionsinformationen zurückgeben
            return {
                'type': 'entity_entity',
                'entity1_id': entity1.get('id', -1),
                'entity2_id': entity2.get('id', -1),
                'position': pos1 + normal * radius1,  # Kollisionspunkt
                'normal': normal,
                'overlap': overlap
            }
        
        # Keine Kollision
        return None

    def detect_collisions(self, entities, terrain=None):
        """
        Erkennt Kollisionen zwischen Entitäten und mit dem Terrain.
        Gibt eine Liste von Kollisionsereignissen zurück.
        """
        collisions = []
        
        # Entitäten-Kollisionen prüfen
        for i, entity1 in enumerate(entities):
            # Terrain-Kollision prüfen
            if terrain is not None:
                terrain_collisions = self._check_terrain_collision(entity1, terrain)
                collisions.extend(terrain_collisions)
            
            # Kollisionen mit anderen Entitäten prüfen
            for j, entity2 in enumerate(entities[i+1:], i+1):
                if i != j:  # Nicht mit sich selbst kollidieren
                    collision = self._check_entity_collision(entity1, entity2)
                    if collision:
                        collisions.append(collision)
        
        return collisions
    
    def resolve_collisions(self, entities, collisions, terrain=None):
        """
        Löst Kollisionen auf, indem Positionen und Geschwindigkeiten angepasst werden.
        """
        for _ in range(self.collision_iterations):
            for collision in collisions:
                if collision['type'] == 'entity_entity':
                    entity1_id = collision['entity1_id']
                    entity2_id = collision['entity2_id']
                    
                    # Suche die Entitäten in der Liste
                    entity1 = None
                    entity2 = None
                    for entity in entities:
                        if entity.get('id') == entity1_id:
                            entity1 = entity
                        elif entity.get('id') == entity2_id:
                            entity2 = entity
                    
                    if entity1 and entity2:
                        self._resolve_entity_collision(entity1, entity2, collision)
                
                elif collision['type'] == 'entity_terrain':
                    entity_id = collision['entity_id']
                    
                    # Suche die Entität in der Liste
                    entity = None
                    for e in entities:
                        if e.get('id') == entity_id:
                            entity = e
                            break
                    
                    if entity and terrain:
                        self._resolve_terrain_collision(entity, collision, terrain)
        
        return entities
    
   
    
    def _check_terrain_collision(self, entity, terrain):
        """
        Überprüft Kollision zwischen einer Entität und dem Terrain.
        """
        collisions = []
        
        position = entity.get('position', np.zeros(3))
        radius = entity.get('collision_radius', 0.5)
        
        # Vereinfachte Höhenabfrage
        terrain_height = terrain.get_height(position[0], position[1])
        
        # Kollision mit dem Boden
        if position[2] - radius < terrain_height:
            normal = np.array([0, 0, 1])  # Nach oben gerichtete Normale
            overlap = terrain_height - (position[2] - radius)
            
            # Terrain-Material an der Kollisionsstelle ermitteln
            terrain_material = terrain.get_material(position[0], position[1])
            
            collisions.append({
                'type': 'entity_terrain',
                'entity_id': entity.get('id', -1),
                'terrain_height': terrain_height,
                'overlap': overlap,
                'normal': normal,
                'position': np.array([position[0], position[1], terrain_height]),
                'material': terrain_material
            })
        
        # Mehr detaillierte Terrain-Kollisionen für höhere Detail-Level
        if self.detail_level >= 3:
            # Mehrere Punkte um die Entität prüfen für genauere Kollisionen mit unebenem Terrain
            check_points = 8 if self.detail_level == 4 else 4
            for i in range(check_points):
                angle = i * (2 * np.pi / check_points)
                offset_x = radius * 0.8 * np.cos(angle)
                offset_y = radius * 0.8 * np.sin(angle)
                
                check_x = position[0] + offset_x
                check_y = position[1] + offset_y
                
                try:
                    terrain_height = terrain.get_height(check_x, check_y)
                    
                    if position[2] - radius < terrain_height:
                        # Berechne lokale Flächennormale für realistischere Kollisionen
                        dx = terrain.get_height(check_x + 0.1, check_y) - terrain.get_height(check_x - 0.1, check_y)
                        dy = terrain.get_height(check_x, check_y + 0.1) - terrain.get_height(check_x, check_y - 0.1)
                        
                        # Normale aus Steigung berechnen
                        normal = np.array([-dx, -dy, 0.2])
                        normal = normal / np.linalg.norm(normal)
                        
                        overlap = terrain_height - (position[2] - radius)
                        
                        terrain_material = terrain.get_material(check_x, check_y)
                        
                        collisions.append({
                            'type': 'entity_terrain',
                            'entity_id': entity.get('id', -1),
                            'terrain_height': terrain_height,
                            'overlap': overlap,
                            'normal': normal,
                            'position': np.array([check_x, check_y, terrain_height]),
                            'material': terrain_material
                        })
                except:
                    # Bei Fehler in der Terrain-Abfrage diese Position überspringen
                    continue
        
        return collisions
    
    def _resolve_entity_collision(self, entity1, entity2, collision):
        """
        Löst eine Kollision zwischen zwei Entitäten auf.
        """
        # Masse und Elastizität der Entitäten
        mass1 = entity1.get('mass', 1.0)
        mass2 = entity2.get('mass', 1.0)
        
        # Elastizität (Restitution) der Kollision
        restitution = (entity1.get('elasticity', self.elasticity) + 
                      entity2.get('elasticity', self.elasticity)) / 2
        
        # Kollisionsauflösung durch Positionskorrektur
        if mass1 + mass2 > 0:
            correction1 = collision['overlap'] * (mass2 / (mass1 + mass2))
            correction2 = collision['overlap'] * (mass1 / (mass1 + mass2))
            
            # Positionen korrigieren
            entity1['position'] -= collision['normal'] * correction1
            entity2['position'] += collision['normal'] * correction2
        
        # Impulsauflösung für realistische Abstoßung
        if self.detail_level >= 2:
            v1 = entity1.get('velocity', np.zeros(3))
            v2 = entity2.get('velocity', np.zeros(3))
            
            # Relativgeschwindigkeit entlang der Kollisionsnormalen
            rv = v2 - v1
            velocity_along_normal = np.dot(rv, collision['normal'])
            
            # Wenn sich die Objekte bereits voneinander wegbewegen, keine Kollisionsauflösung
            if velocity_along_normal > 0:
                return
            
            # Impulsberechnung
            j = -(1 + restitution) * velocity_along_normal
            j /= (1/mass1 + 1/mass2)
            
            # Impuls anwenden
            impulse = j * collision['normal']
            entity1['velocity'] -= impulse / mass1
            entity2['velocity'] += impulse / mass2
            
            # Reibung anwenden
            if self.detail_level >= 3:
                # Tangente zur Kollisionsebene
                tangent = rv - (np.dot(rv, collision['normal']) * collision['normal'])
                tangent_norm = np.linalg.norm(tangent)
                
                if tangent_norm > 1e-6:
                    tangent = tangent / tangent_norm
                    
                    # Reibungskoeffizient
                    friction = (entity1.get('friction', 0.5) + entity2.get('friction', 0.5)) / 2
                    
                    # Reibungsimpuls
                    j_t = -np.dot(rv, tangent)
                    j_t /= (1/mass1 + 1/mass2)
                    
                    # Coulomb-Reibungsmodell
                    j_t = max(min(j_t, friction * j), -friction * j)
                    
                    # Reibungsimpuls anwenden
                    friction_impulse = j_t * tangent
                    entity1['velocity'] -= friction_impulse / mass1
                    entity2['velocity'] += friction_impulse / mass2
    
    def _resolve_terrain_collision(self, entity, collision, terrain):
        """
        Löst eine Kollision zwischen einer Entität und dem Terrain auf.
        """
        # Korrigiere die Position des Objekts, um Überlappung zu beseitigen
        entity['position'] += collision['normal'] * collision['overlap']
        
        # Geschwindigkeitsänderung berechnen
        if self.detail_level >= 2:
            velocity = entity.get('velocity', np.zeros(3))
            
            # Material-Eigenschaften
            material = self.materials.get(collision['material'], self.materials['concrete'])
            elasticity = (entity.get('elasticity', self.elasticity) + material['elasticity']) / 2
            friction = (entity.get('friction', 0.5) + material['roughness']) / 2
            
            # Relativgeschwindigkeit entlang der Kollisionsnormalen
            velocity_along_normal = np.dot(velocity, collision['normal'])
            
            # Wenn sich die Entität bereits vom Terrain wegbewegt, keine Kollisionsauflösung
            if velocity_along_normal > 0:
                return
            
            # Normale Komponente der Geschwindigkeit umkehren (Aufprall)
            normal_velocity = velocity_along_normal * collision['normal']
            entity['velocity'] -= (1 + elasticity) * normal_velocity
            
            # Reibung anwenden (tangentiale Komponente der Geschwindigkeit reduzieren)
            if self.detail_level >= 3:
                tangential_velocity = velocity - normal_velocity
                tangential_speed = np.linalg.norm(tangential_velocity)
                
                if tangential_speed > 1e-6:
                    # Richtung der tangentialen Geschwindigkeit
                    tangent = tangential_velocity / tangential_speed
                    
                    # Reibungskraft berechnen
                    friction_magnitude = min(friction * np.abs(velocity_along_normal), tangential_speed)
                    friction_velocity = friction_magnitude * tangent
                    
                    # Geschwindigkeit durch Reibung reduzieren
                    entity['velocity'] -= friction_velocity


# Terrain-Generator und -Manager
class Terrain:
    """
    Generiert und verwaltet das Terrain für die Simulation.
    """
    def __init__(self, size=(100, 100), resolution=1.0, detail_level=2):
        self.size = size  # (width, height) in Metern
        self.resolution = resolution  # Auflösung in Metern pro Pixel
        self.detail_level = detail_level
        
        # Berechne die Größe des Höhenfeldes basierend auf Größe und Auflösung
        self.height_map_size = (int(size[0] / resolution), int(size[1] / resolution))
        
        # Initialisiere Höhenkarte und Materialkarte
        self.height_map = np.zeros(self.height_map_size)
        self.material_map = np.zeros(self.height_map_size, dtype=np.int32)
        
        # Material-IDs und Eigenschaften
        self.materials = {
            0: 'concrete',
            1: 'grass',
            2: 'sand',
            3: 'ice',
            4: 'metal',
            5: 'rubber',
            6: 'wood'
        }
        
        # Generiere das Terrain
        self._generate_terrain()
    
    def _generate_terrain(self):
        """
        Generiert ein realistisches Terrain mit verschiedenen Höhen und Materialien.
        """
        # Parameter für das Perlin-Rauschen
        octaves = {
            1: 3,
            2: 4,
            3: 5,
            4: 6
        }[self.detail_level]
        
        persistence = 0.5
        lacunarity = 2.0
        
        # Höhenkarte mit Perlin-Rauschen generieren
        height_noise = self._generate_perlin_noise(self.height_map_size, octaves, persistence, lacunarity)
        
        # Höhen skalieren und anpassen
        height_scale = 5.0  # Maximale Höhe in Metern
        height_offset = -2.0  # Minimale Höhe
        
        self.height_map = height_noise * height_scale + height_offset
        
        # Flaches Gebiet für den Start erzeugen
        center_x = self.height_map_size[0] // 2
        center_y = self.height_map_size[1] // 2
        radius = min(self.height_map_size) // 10
        
        for y in range(self.height_map_size[1]):
            for x in range(self.height_map_size[0]):
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if distance < radius:
                    # Flachen Boden im Zentrum erstellen
                    smooth_factor = 1.0 - max(0, min(1, (distance / radius)))
                    self.height_map[y, x] = self.height_map[y, x] * (1 - smooth_factor) + 0.0 * smooth_factor
        
        # Glätten
        if self.detail_level >= 2:
            self.height_map = gaussian_filter(self.height_map, sigma=1.0)
        
        # Materialkarte basierend auf Höhe und Steigung generieren
        gradient_x = np.gradient(self.height_map, axis=1)
        gradient_y = np.gradient(self.height_map, axis=0)
        slope = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Materialkarten-Initialisierung
        self.material_map.fill(1)  # Standard: Gras
        
        # Materialzuweisung basierend auf Höhe und Steigung
        for y in range(self.height_map_size[1]):
            for x in range(self.height_map_size[0]):
                height = self.height_map[y, x]
                current_slope = slope[y, x]
                
                if height < -1.5:
                    # Tiefe Bereiche: Sand
                    self.material_map[y, x] = 2
                elif height > 3.0:
                    # Hohe Bereiche
                    if current_slope > 0.7:
                        # Steile Bereiche: Fels (Beton)
                        self.material_map[y, x] = 0
                    else:
                        # Flache hohe Bereiche: Gras
                        self.material_map[y, x] = 1
                else:
                    # Mittlere Höhen
                    if current_slope > 0.5:
                        # Steile mittlere Bereiche: Fels
                        self.material_map[y, x] = 0
                    elif current_slope < 0.1 and np.random.random() < 0.2:
                        # Sehr flache Bereiche: Zufällig Eis platzieren
                        self.material_map[y, x] = 3
        
        # Spezielle Bereiche
        if self.detail_level >= 3:
            # Metallplattform
            platform_x = self.height_map_size[0] // 4
            platform_y = self.height_map_size[1] // 4
            platform_size = min(self.height_map_size) // 15
            
            for y in range(platform_y - platform_size, platform_y + platform_size):
                for x in range(platform_x - platform_size, platform_x + platform_size):
                    if 0 <= x < self.height_map_size[0] and 0 <= y < self.height_map_size[1]:
                        distance = max(abs(x - platform_x), abs(y - platform_y))
                        if distance < platform_size - 2:
                            # Metallplattform mit einheitlicher Höhe
                            self.height_map[y, x] = 0.5
                            self.material_map[y, x] = 4  # Metall
            
            # Pfade aus Gummi
            path_width = max(1, min(self.height_map_size) // 50)
            
            # Horizontaler Pfad
            path_y = self.height_map_size[1] // 2
            for y in range(path_y - path_width, path_y + path_width):
                for x in range(0, self.height_map_size[0]):
                    if 0 <= y < self.height_map_size[1]:
                        # Pfad folgt dem Terrain, aber mit leicht geglätteter Höhe
                        if self.detail_level == 4:
                            # Glätte den Pfad für bessere Befahrbarkeit
                            neighbors = []
                            for ny in range(max(0, y-2), min(self.height_map_size[1], y+3)):
                                for nx in range(max(0, x-2), min(self.height_map_size[0], x+3)):
                                    neighbors.append(self.height_map[ny, nx])
                            self.height_map[y, x] = np.median(neighbors)
                        self.material_map[y, x] = 5  # Gummi
    
    def _generate_perlin_noise(self, size, octaves, persistence, lacunarity):
        """
        Generiert Perlin-Rauschen für natürlich aussehende Landschaften.
        """
        # Vereinfachte Perlin-Rauschen-Implementierung für natürlichere Landschaften
        noise_map = np.zeros(size)
        
        # Seed für die Zufallszahlen
        np.random.seed(42)
        
        # Erstelle Basisrauschen für verschiedene Frequenzen
        base_noise = []
        for _ in range(octaves):
            # Erzeuge ein zufälliges Rauschen mit der richtigen Größe
            freq_noise = np.random.rand(*size)
            # Glätten für bessere Ergebnisse
            freq_noise = gaussian_filter(freq_noise, sigma=1.0)
            base_noise.append(freq_noise)
        
        # Kombiniere das Rauschen der verschiedenen Frequenzen
        amplitude = 1.0
        frequency = 1.0
        for o in range(octaves):
            # Skaliere das Basisrauschen entsprechend der aktuellen Frequenz
            scaled_size = (
                int(size[0] * frequency),
                int(size[1] * frequency)
            )
            
            # Wenn die skalierte Größe zu klein ist, überspringen
            if scaled_size[0] <= 0 or scaled_size[1] <= 0:
                continue
            
            # Größe anpassen und interpolieren
            if scaled_size != size:
                # Dimensionen begrenzen, um Indizierungsfehler zu vermeiden
                max_x = min(scaled_size[0], base_noise[o].shape[0])
                max_y = min(scaled_size[1], base_noise[o].shape[1])
                
                # Skaliere das Rauschen
                sampled_noise = base_noise[o][:max_x, :max_y]
                
                # Vergrößern auf die Originalgröße mit bilinearer Interpolation
                from scipy.ndimage import zoom
                try:
                    zoom_factors = (size[0] / max_x, size[1] / max_y)
                    scaled_noise = zoom(sampled_noise, zoom_factors, order=1)
                    
                    # Anpassen, um Größenfehler zu vermeiden
                    scaled_noise = scaled_noise[:size[0], :size[1]]
                except:
                    # Fallback bei Fehlern
                    scaled_noise = np.random.rand(*size)
            else:
                scaled_noise = base_noise[o]
            
            # Rauschen mit aktueller Amplitude zum Gesamtrauschen addieren
            noise_map += scaled_noise * amplitude
            
            # Amplitude und Frequenz für die nächste Oktave aktualisieren
            amplitude *= persistence
            frequency *= lacunarity
        
        # Normalisieren auf den Bereich [0, 1]
        noise_map = (noise_map - np.min(noise_map)) / (np.max(noise_map) - np.min(noise_map))
        
        return noise_map
    
    def get_height(self, x, y):
        """
        Gibt die Höhe des Terrains an den gegebenen Weltkoordinaten zurück.
        """
        try:
            # Konvertiere Weltkoordinaten in Indizes
            x_idx = int((x + self.size[0]/2) / self.resolution)
            y_idx = int((y + self.size[1]/2) / self.resolution)
            
            # Begrenze auf gültige Indizes
            x_idx = max(0, min(x_idx, self.height_map_size[0] - 1))
            y_idx = max(0, min(y_idx, self.height_map_size[1] - 1))
            
            # Bilineare Interpolation für glattere Höhen (bei höheren Detail-Leveln)
            if self.detail_level >= 3:
                # Fraktionaler Anteil
                x_frac = (x + self.size[0]/2) / self.resolution - x_idx
                y_frac = (y + self.size[1]/2) / self.resolution - y_idx
                
                # Indizes der vier Nachbarpunkte
                x0 = max(0, min(x_idx, self.height_map_size[0] - 1))
                x1 = max(0, min(x_idx + 1, self.height_map_size[0] - 1))
                y0 = max(0, min(y_idx, self.height_map_size[1] - 1))
                y1 = max(0, min(y_idx + 1, self.height_map_size[1] - 1))
                
                # Höhen der vier Nachbarpunkte
                h00 = self.height_map[y0, x0]
                h01 = self.height_map[y0, x1]
                h10 = self.height_map[y1, x0]
                h11 = self.height_map[y1, x1]
                
                # Bilineare Interpolation
                return (h00 * (1 - x_frac) * (1 - y_frac) +
                        h01 * x_frac * (1 - y_frac) +
                        h10 * (1 - x_frac) * y_frac +
                        h11 * x_frac * y_frac)
            else:
                # Einfache nicht-interpolierte Höhenabfrage
                return self.height_map[y_idx, x_idx]
        except:
            # Fallback bei Fehlern
            return 0.0
    
    def get_material(self, x, y):
        """
        Gibt das Material des Terrains an den gegebenen Weltkoordinaten zurück.
        """
        try:
            # Konvertiere Weltkoordinaten in Indizes
            x_idx = int((x + self.size[0]/2) / self.resolution)
            y_idx = int((y + self.size[1]/2) / self.resolution)
            
            # Begrenze auf gültige Indizes
            x_idx = max(0, min(x_idx, self.height_map_size[0] - 1))
            y_idx = max(0, min(y_idx, self.height_map_size[1] - 1))
            
            # Material-ID abrufen und in Materialname umwandeln
            material_id = self.material_map[y_idx, x_idx]
            return self.materials.get(material_id, 'concrete')
        except:
            # Fallback bei Fehlern
            return 'concrete'
    
    def render_2d(self, ax=None):
        """
        Rendert eine 2D-Darstellung des Terrains.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Höhenkarte als Konturplot darstellen
        x = np.linspace(-self.size[0]/2, self.size[0]/2, self.height_map_size[0])
        y = np.linspace(-self.size[1]/2, self.size[1]/2, self.height_map_size[1])
        X, Y = np.meshgrid(x, y)
        
        # Höhenkarte mit Farbverlauf darstellen
        contour = ax.contourf(X, Y, self.height_map, levels=20, cmap='terrain')
        plt.colorbar(contour, ax=ax, label='Höhe (m)')
        
        # Material-Visualisierung als transparente Überlagerung
        material_colors = {
            0: (0.5, 0.5, 0.5, 0.3),  # Beton (grau)
            1: (0.0, 0.8, 0.0, 0.3),  # Gras (grün)
            2: (0.9, 0.9, 0.0, 0.3),  # Sand (gelb)
            3: (0.8, 0.8, 1.0, 0.3),  # Eis (hellblau)
            4: (0.6, 0.6, 0.8, 0.3),  # Metall (blaugrau)
            5: (0.3, 0.3, 0.3, 0.3),  # Gummi (dunkelgrau)
            6: (0.6, 0.3, 0.0, 0.3)   # Holz (braun)
        }
        
        if self.detail_level >= 3:
            # Bei höherem Detaillevel die Materialien als farbige Punkte anzeigen
            for y in range(0, self.height_map_size[1], 5):
                for x in range(0, self.height_map_size[0], 5):
                    material_id = self.material_map[y, x]
                    color = material_colors.get(material_id, (0, 0, 0, 0))
                    ax.scatter(x * self.resolution - self.size[0]/2, 
                              y * self.resolution - self.size[1]/2, 
                              color=color, s=10)
        
        ax.set_title('Terrain-Höhenkarte')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        
        return ax
    
    def render_3d(self, ax=None, resolution_factor=1):
        """
        Rendert eine 3D-Darstellung des Terrains.
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        # Bei hohem Detaillevel die volle Auflösung verwenden, sonst reduzieren
        downscale = 4 // resolution_factor
        downscale = max(1, downscale)
        
        # Reduziere die Auflösung für die Darstellung
        x = np.linspace(-self.size[0]/2, self.size[0]/2, self.height_map_size[0]//downscale)
        y = np.linspace(-self.size[1]/2, self.size[1]/2, self.height_map_size[1]//downscale)
        X, Y = np.meshgrid(x, y)
        
        # Reduziere die Höhenkarte für die Darstellung
        Z = self.height_map[::downscale, ::downscale]
        
        # Materialkarte für die Farbgebung reduzieren
        material_downscaled = self.material_map[::downscale, ::downscale]
        
        # Farbkarte basierend auf Materialien erstellen
        material_colors = {
            0: (0.5, 0.5, 0.5),  # Beton (grau)
            1: (0.0, 0.8, 0.0),  # Gras (grün)
            2: (0.9, 0.9, 0.0),  # Sand (gelb)
            3: (0.8, 0.8, 1.0),  # Eis (hellblau)
            4: (0.6, 0.6, 0.8),  # Metall (blaugrau)
            5: (0.3, 0.3, 0.3),  # Gummi (dunkelgrau)
            6: (0.6, 0.3, 0.0)   # Holz (braun)
        }
        
        # Farbarray erstellen
        colors = np.zeros((Z.shape[0], Z.shape[1], 3))
        for y in range(material_downscaled.shape[0]):
            for x in range(material_downscaled.shape[1]):
                material_id = material_downscaled[y, x]
                colors[y, x] = material_colors.get(material_id, (0, 0, 0))
        
        # Surface plot mit Farben basierend auf Materialien
        surf = ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, 
                              antialiased=True, shade=True)
        
        # Achsenbeschriftungen
        ax.set_title('Terrain 3D-Ansicht')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Höhe (m)')
        
        # Sichtbereich anpassen
        ax.set_zlim(np.min(self.height_map) - 1, np.max(self.height_map) + 1)
        
        # Blickwinkel einstellen
        ax.view_init(elev=30, azim=45)
        
        return ax


# Erweiterte Sensormodellierung mit realistischem Rauschen und Verzögerungen
class SensorSystem:
    """
    Umfassendes Sensorsystem für den Roboter, das verschiedene Sensortypen simuliert.
    """
    def __init__(self, config=None):
        """
        Initialisiert das Sensorsystem mit der gegebenen Konfiguration.
        
        config: Dictionary mit Sensorparametern wie Auflösung, Rauschen usw.
        """
        # Standardkonfiguration
        self.config = {
            'camera': {
                'resolution': (64, 64),
                'fov': 90,  # Sichtfeld in Grad
                'range': 20.0,  # Sichtweite in Metern
                'noise_level': 0.02,
                'update_rate': 30,  # Hz
                'delay': 0.01  # Verzögerung in Sekunden
            },
            'lidar': {
                'num_rays': 16,  # Anzahl der Strahlen
                'fov': 360,  # Vollständiger Kreis
                'range': 30.0,
                'noise_level': 0.01,
                'update_rate': 20,
                'delay': 0.02
            },
            'imu': {
                'noise_level': 0.005,
                'drift_level': 0.001,
                'update_rate': 100,
                'delay': 0.001
            },
            'touch': {
                'num_sensors': 8,
                'sensitivity': 0.8,
                'noise_level': 0.01,
                'update_rate': 50,
                'delay': 0.005
            },
            'smell': {
                'num_sensors': 8,
                'sensitivity': 0.7,
                'range': 15.0,
                'noise_level': 0.05,
                'update_rate': 10,
                'delay': 0.05
            },
            'gps': {
                'noise_level': 0.5,  # Meter
                'update_rate': 5,
                'delay': 0.1
            }
        }
        
        # Update mit benutzerdefinierten Einstellungen
        if config is not None:
            self._update_config(config)
        
        # Sensorzustandsspeicher
        self.last_update_time = {sensor: 0.0 for sensor in self.config}
        self.last_sensor_data = {sensor: None for sensor in self.config}
        self.sensor_queues = {sensor: queue.Queue() for sensor in self.config}
        
        # Initialisiere alle Sensoren
        self._init_sensors()
    
    def _update_config(self, config):
        """
        Aktualisiert die Konfiguration mit benutzerdefinierten Einstellungen.
        """
        for sensor_type, sensor_config in config.items():
            if sensor_type in self.config:
                for key, value in sensor_config.items():
                    self.config[sensor_type][key] = value
    
    def _init_sensors(self):
        """
        Initialisiert alle Sensoren basierend auf der Konfiguration.
        """
        # Kamera-Initialisierung
        self.camera_fov_rad = np.radians(self.config['camera']['fov'])
        
        # LiDAR-Initialisierung
        self.lidar_angles = np.linspace(0, 2*np.pi, self.config['lidar']['num_rays'], endpoint=False)
        
        # Initialisierung des IMU mit Bias
        self.imu_bias = {
            'accel': np.random.normal(0, self.config['imu']['drift_level'], 3),
            'gyro': np.random.normal(0, self.config['imu']['drift_level'], 3)
        }
        
        # Initialisiere GPS mit zufälligem Offset
        self.gps_offset = np.random.normal(0, self.config['gps']['noise_level'] / 3, 2)
        
        # Berührungs- und Geruchssensoren
        self.touch_sensor_positions = self._calculate_sensor_positions(
            self.config['touch']['num_sensors']
        )
        
        self.smell_sensor_positions = self._calculate_sensor_positions(
            self.config['smell']['num_sensors']
        )
    
    def _calculate_sensor_positions(self, num_sensors):
        """
        Berechnet die Positionen für eine gleichmäßige Verteilung von Sensoren um den Roboter.
        """
        angles = np.linspace(0, 2*np.pi, num_sensors, endpoint=False)
        positions = []
        
        for angle in angles:
            # Position relativ zum Robotermittelpunkt
            pos = np.array([np.cos(angle), np.sin(angle), 0])
            positions.append(pos)
        
        return positions
    
    def update(self, time_now, robot_state, environment):
        """
        Aktualisiert alle Sensordaten basierend auf der aktuellen Zeit und dem Roboterzustand.
        """
        # Für jeden Sensortyp prüfen, ob ein Update fällig ist
        for sensor_type, config in self.config.items():
            update_interval = 1.0 / config['update_rate']
            
            if time_now - self.last_update_time[sensor_type] >= update_interval:
                # Sensordaten generieren
                sensor_data = self._generate_sensor_data(sensor_type, robot_state, environment)
                
                # Verzögerung simulieren: Daten in die Warteschlange stellen, um sie später abzurufen
                self.sensor_queues[sensor_type].put({
                    'data': sensor_data,
                    'time': time_now,
                    'available_at': time_now + config['delay']
                })
                
                self.last_update_time[sensor_type] = time_now
        
        # Prüfe, ob verzögerte Sensorwerte jetzt verfügbar sind
        self._process_sensor_queues(time_now)
    
    def _process_sensor_queues(self, time_now):
        """
        Verarbeitet die Sensorwarteschlangen und macht Daten verfügbar, wenn ihre Verzögerung abgelaufen ist.
        """
        for sensor_type, sensor_queue in self.sensor_queues.items():
            # Prüfe alle ausstehenden Sensoraktualisierungen
            while not sensor_queue.empty():
                # Schaue das vorderste Element an, ohne es zu entfernen
                next_update = sensor_queue.queue[0]
                
                if next_update['available_at'] <= time_now:
                    # Diese Daten können jetzt verfügbar gemacht werden
                    update = sensor_queue.get()
                    self.last_sensor_data[sensor_type] = update['data']
                else:
                    # Noch nicht bereit, also die Schleife verlassen
                    break
    
    def _generate_sensor_data(self, sensor_type, robot_state, environment):
        """
        Generiert realistische Sensordaten für den angegebenen Sensortyp.
        """
        position = robot_state.get('position', np.zeros(3))
        orientation = robot_state.get('rotation', np.zeros(3))
        orientation_quat = robot_state.get('rotation_quat', np.array([1.0, 0.0, 0.0, 0.0]))
        velocity = robot_state.get('velocity', np.zeros(3))
        angular_velocity = robot_state.get('angular_velocity', np.zeros(3))
        
        if sensor_type == 'camera':
            return self._generate_camera_data(position, orientation, environment)
        elif sensor_type == 'lidar':
            return self._generate_lidar_data(position, orientation, environment)
        elif sensor_type == 'imu':
            return self._generate_imu_data(position, orientation, velocity, angular_velocity)
        elif sensor_type == 'touch':
            return self._generate_touch_data(position, orientation, environment)
        elif sensor_type == 'smell':
            return self._generate_smell_data(position, orientation, environment)
        elif sensor_type == 'gps':
            return self._generate_gps_data(position)
        else:
            return None
    def _generate_camera_data(self, position, orientation, environment):
        """
        Generiert ein synthetisches Kamerabild.
        """
        # Kamerakonfiguration
        width, height = self.config['camera']['resolution']
        fov = self.camera_fov_rad
        max_range = self.config['camera']['range']
        noise_level = self.config['camera']['noise_level']
        
        # Erstelle ein leeres Bild (RGB)
        image = np.zeros((height, width, 3))
        
        # Berechne die Blickrichtung der Kamera basierend auf der Roboterausrichtung
        yaw = orientation[2]  # Z-Achse Rotation
        
        # Richtungsvektoren für Kamera
        forward = np.array([np.cos(yaw), np.sin(yaw), 0])
        right = np.array([np.cos(yaw + np.pi/2), np.sin(yaw + np.pi/2), 0])
        up = np.array([0, 0, 1])
        
        # Erzeuge Strahlen für jeden Pixel
        for y in range(height):
            for x in range(width):
                # Berechne Richtung für diesen Pixel
                # Konvertiere Pixel zu normalisierten Gerätekoordinaten (-1 bis 1)
                ndc_x = 2 * (x + 0.5) / width - 1
                ndc_y = 1 - 2 * (y + 0.5) / height  # Y ist in Bildkoordinaten umgekehrt
                
                # Berechne Richtungsvektor
                pixel_right = right * ndc_x * np.tan(fov / 2)
                pixel_up = up * ndc_y * np.tan(fov / 2) * (height / width)
                ray_dir = forward + pixel_right + pixel_up
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                
                # Führe einen Raycast durch, um zu sehen, was dieser Pixel "sieht"
                hit, color = self._raycast(position, ray_dir, max_range, environment)
                
                if hit:
                    image[y, x] = color
        
        # Rauschen hinzufügen
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, image.shape)
            image = np.clip(image + noise, 0, 1)
        
        return image
    
    def _generate_lidar_data(self, position, orientation, environment):
        """
        Generiert LiDAR-Daten (Entfernungsmessungen in verschiedenen Richtungen).
        """
        # LiDAR-Konfiguration
        num_rays = self.config['lidar']['num_rays']
        max_range = self.config['lidar']['range']
        noise_level = self.config['lidar']['noise_level']
        
        # Array für Distanzmessungen
        distances = np.full(num_rays, max_range)
        
        # Berechne die Blickrichtung basierend auf der Roboterausrichtung
        yaw = orientation[2]  # Z-Achse Rotation
        
        # Für jeden LiDAR-Strahl
        for i, angle in enumerate(self.lidar_angles):
            # Berechne die globale Richtung dieses Strahls
            global_angle = angle + yaw
            ray_dir = np.array([np.cos(global_angle), np.sin(global_angle), 0])
            
            # Führe einen Raycast durch
            hit, dist = self._lidar_raycast(position, ray_dir, max_range, environment)
            
            if hit:
                distances[i] = dist
        
        # Rauschen hinzufügen
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * max_range, num_rays)
            distances = np.clip(distances + noise, 0, max_range)
        
        return distances
    
    def _generate_imu_data(self, position, orientation, velocity, angular_velocity):
        """
        Generiert IMU-Daten (Beschleunigung und Winkelgeschwindigkeit).
        """
        # IMU-Konfiguration
        noise_level = self.config['imu']['noise_level']
        
        # Berechne Beschleunigung aus der Änderung der Geschwindigkeit
        # Für eine reale Simulation bräuchten wir die vorherige Geschwindigkeit
        # Hier vereinfachen wir und nehmen die aktuelle Geschwindigkeit als Basis
        acceleration = np.zeros(3)
        if hasattr(self, 'prev_velocity'):
            # Schätze Beschleunigung aus Geschwindigkeitsänderung
            dt = 1.0 / self.config['imu']['update_rate']
            acceleration = (velocity - self.prev_velocity) / max(dt, 1e-6)
        
        # Speichere aktuelle Geschwindigkeit für die nächste Aktualisierung
        self.prev_velocity = velocity.copy()
        
        # Gravitation hinzufügen (in globalen Koordinaten ist sie immer in Z-Richtung)
        global_gravity = np.array([0, 0, -9.81])
        
        # Konvertiere in lokale Koordinaten des Roboters
        # In einer detaillierteren Simulation würden wir eine Rotationsmatrix verwenden
        # Hier verwenden wir eine vereinfachte Version
        roll, pitch, yaw = orientation
        
        # Erstelle Rotationsmatrix
        # Hinweis: Dies ist eine vereinfachte Darstellung der Rotation
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # Kombinierte Rotationsmatrix (Reihenfolge: Yaw -> Pitch -> Roll)
        R = R_x @ R_y @ R_z
        
        # Inverse Rotationsmatrix, um von global nach lokal zu konvertieren
        R_inv = R.T
        
        # Konvertiere globale Gravitation in lokale Koordinaten
        local_gravity = R_inv @ global_gravity
        
        # IMU misst die Beschleunigung in lokalen Koordinaten ohne Gravitation
        # In der Realität enthält sie Gravitation, aber für die Simulation trennen wir diese
        local_acceleration = R_inv @ acceleration
        
        # Gesamtbeschleunigung, die der IMU "sieht" (mit Gravitation)
        measured_acceleration = local_acceleration + local_gravity
        
        # Füge Bias und Rauschen hinzu
        accel_noise = np.random.normal(0, noise_level, 3)
        gyro_noise = np.random.normal(0, noise_level, 3)
        
        # Messwerte mit Bias und Rauschen
        measured_acceleration += self.imu_bias['accel'] + accel_noise
        measured_angular_velocity = angular_velocity + self.imu_bias['gyro'] + gyro_noise
        
        # Aktualisiere den Bias leicht (Drift simulieren)
        drift_level = self.config['imu']['drift_level']
        self.imu_bias['accel'] += np.random.normal(0, drift_level, 3) * (1.0 / self.config['imu']['update_rate'])
        self.imu_bias['gyro'] += np.random.normal(0, drift_level, 3) * (1.0 / self.config['imu']['update_rate'])
        
        return {
            'acceleration': measured_acceleration,
            'angular_velocity': measured_angular_velocity
        }
    
    def _generate_touch_data(self, position, orientation, environment):
        """
        Generiert Daten für Berührungssensoren.
        """
        # Sensor-Konfiguration
        num_sensors = self.config['touch']['num_sensors']
        sensitivity = self.config['touch']['sensitivity']
        noise_level = self.config['touch']['noise_level']
        
        # Array für Berührungsstärke (0 = keine Berührung, 1 = maximale Berührung)
        touch_values = np.zeros(num_sensors)
        
        # Rotationsmatrix basierend auf der Roboterausrichtung
        yaw = orientation[2]  # Z-Achse Rotation
        R = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Für jeden Berührungssensor
        for i, sensor_pos_local in enumerate(self.touch_sensor_positions):
            # Konvertiere lokale Sensorposition in globale Koordinaten
            sensor_pos_global = position + R @ sensor_pos_local
            
            # Prüfe Berührungen mit Objekten in der Umgebung
            # Vereinfachte Version: Suche nach nahen Objekten und berechne Berührungsstärke
            try:
                for entity in environment.get_entities_near(sensor_pos_global, 0.5):
                    # Abstand zum Objekt
                    distance = np.linalg.norm(entity['position'] - sensor_pos_global)
                    
                    # Wenn nahe genug, registriere eine Berührung
                    if distance <= entity.get('collision_radius', 0.1) + 0.1:
                        # Skaliere die Stärke mit dem Abstand (näher = stärker)
                        strength = (1.0 - distance / (entity.get('collision_radius', 0.1) + 0.1)) * sensitivity
                        touch_values[i] = max(touch_values[i], strength)
            except:
                # Bei Fehlern in der Entitätsabfrage überspringen
                pass
            
            # Prüfe Berührung mit dem Terrain
            try:
                if environment.terrain is not None:
                    terrain_height = environment.terrain.get_height(sensor_pos_global[0], sensor_pos_global[1])
                    if sensor_pos_global[2] <= terrain_height + 0.1:
                        # Skaliere Stärke mit der Eindringtiefe
                        penetration = max(0, terrain_height + 0.1 - sensor_pos_global[2])
                        terrain_strength = min(1.0, penetration * 10) * sensitivity
                        touch_values[i] = max(touch_values[i], terrain_strength)
            except:
                # Bei Fehlern in der Terrainabfrage überspringen
                pass
        
        # Rauschen hinzufügen
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, num_sensors)
            touch_values = np.clip(touch_values + noise, 0, 1)
        
        return touch_values
    
    def _generate_smell_data(self, position, orientation, environment):
        """
        Generiert Daten für Geruchssensoren (simuliert chemische Sensoren).
        """
        # Sensor-Konfiguration
        num_sensors = self.config['smell']['num_sensors']
        sensitivity = self.config['smell']['sensitivity']
        max_range = self.config['smell']['range']
        noise_level = self.config['smell']['noise_level']
        
        # Array für Geruchsstärke und -typ
        # Jeder Sensor liefert ein Paar von Werten: [Stärke, Typ]
        smell_values = np.zeros((num_sensors, 2))
        
        # Rotationsmatrix basierend auf der Roboterausrichtung
        yaw = orientation[2]  # Z-Achse Rotation
        R = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Geruchsquellen aus der Umgebung holen
        try:
            smell_sources = environment.get_smell_sources()
        except:
            # Fallback wenn keine Geruchsquellen verfügbar sind
            smell_sources = []
        
        # Für jeden Geruchssensor
        for i, sensor_pos_local in enumerate(self.smell_sensor_positions):
            # Konvertiere lokale Sensorposition in globale Koordinaten
            sensor_pos_global = position + R @ sensor_pos_local
            
            # Für jede Geruchsquelle berechnen, wie stark sie an diesem Sensor wahrgenommen wird
            for source in smell_sources:
                source_pos = source['position']
                source_intensity = source['intensity']
                source_type = source['type']  # z.B. 0=Ziel, 1=Hindernis, 2=Gefahr, usw.
                
                # Abstand zur Quelle
                distance = np.linalg.norm(source_pos - sensor_pos_global)
                
                # Wenn innerhalb der Reichweite
                if distance <= max_range:
                    # Intensität basierend auf inverser quadratischer Abnahme
                    intensity = source_intensity / (1 + distance**2)
                    
                    # Richtungsabhängigkeit simulieren
                    # Gerüche sind in Windrichtung stärker wahrnehmbar
                    try:
                        wind_direction = environment.get_wind_direction()
                    except:
                        wind_direction = None
                    
                    if wind_direction is not None:
                        # Vektor von der Quelle zum Sensor
                        source_to_sensor = sensor_pos_global - source_pos
                        source_to_sensor = source_to_sensor / np.linalg.norm(source_to_sensor)
                        
                        # Skalarprodukt gibt an, wie sehr der Wind in die gleiche Richtung bläst
                        # (1 = exakt in Richtung Sensor, -1 = exakt weg vom Sensor)
                        wind_alignment = np.dot(wind_direction, source_to_sensor)
                        
                        # Skaliere Intensität basierend auf Windausrichtung
                        # Bei wind_alignment=-1 wird die Intensität reduziert, bei wind_alignment=1 erhöht
                        wind_factor = 0.2 + 0.8 * (wind_alignment * 0.5 + 0.5)  # Skalieren auf [0.2, 1.0]
                        intensity *= wind_factor
                    
                    # Intensität mit Sensor-Sensitivität skalieren
                    intensity *= sensitivity
                    
                    # Wenn diese Geruchsquelle stärker ist als bisherige Quellen, überschreibe die Werte
                    if intensity > smell_values[i, 0]:
                        smell_values[i, 0] = intensity
                        smell_values[i, 1] = source_type
        
        # Rauschen hinzufügen
        if noise_level > 0:
            intensity_noise = np.random.normal(0, noise_level, (num_sensors, 1))
            # Nur Intensität, nicht Typ, mit Rauschen versehen
            smell_values[:, 0:1] = np.clip(smell_values[:, 0:1] + intensity_noise, 0, 1)
        
        return smell_values
    
    def _generate_gps_data(self, position):
        """
        Generiert simulierte GPS-Daten.
        """
        # GPS-Konfiguration
        noise_level = self.config['gps']['noise_level']
        
        # Reale GPS liefert nur horizontale Position (x, y)
        gps_position = position[:2]
        
        # Rauschen und Offset hinzufügen für realistischere Simulation
        noise = np.random.normal(0, noise_level, 2)
        
        # Das konstante Offset simuliert systematische Fehler
        gps_position = gps_position + noise + self.gps_offset
        
        return gps_position
    
    def _raycast(self, origin, direction, max_distance, environment):
        """
        Führt einen Raycast durch und gibt die Farbe des getroffenen Objekts zurück.
        """
        # Standardfarbe für den Himmel
        sky_color = np.array([0.5, 0.7, 1.0])
        
        # Prüfe Terrain-Kollision
        try:
            if environment.terrain is not None:
                # Vereinfachter Raycast mit dem Terrain
                hit_point, hit_distance = self._raycast_terrain(origin, direction, max_distance, environment.terrain)
                
                if hit_point is not None:
                    # Farbe basierend auf Terrain-Material
                    material = environment.terrain.get_material(hit_point[0], hit_point[1])
                    
                    # Material zu Farbe zuordnen
                    if material == 'concrete':
                        return True, np.array([0.5, 0.5, 0.5])
                    elif material == 'grass':
                        return True, np.array([0.0, 0.8, 0.0])
                    elif material == 'sand':
                        return True, np.array([0.9, 0.9, 0.0])
                    elif material == 'ice':
                        return True, np.array([0.8, 0.8, 1.0])
                    elif material == 'metal':
                        return True, np.array([0.6, 0.6, 0.8])
                    elif material == 'rubber':
                        return True, np.array([0.3, 0.3, 0.3])
                    elif material == 'wood':
                        return True, np.array([0.6, 0.3, 0.0])
                    else:
                        return True, np.array([0.5, 0.5, 0.5])  # Fallback-Farbe
        except:
            pass  # Bei Fehlern in der Terrain-Prüfung fortfahren
        
        # Prüfe Kollision mit Entitäten
        try:
            for entity in environment.get_visible_entities(origin, direction, max_distance):
                # Vereinfachter Raycast mit kugelförmigen Objekten
                hit, distance = self._raycast_sphere(origin, direction, entity['position'], entity.get('collision_radius', 0.5))
                
                if hit and distance <= max_distance:
                    # Farbe basierend auf Entitätstyp
                    entity_type = entity.get('type', 'default')
                    
                    if entity_type == 'target':
                        return True, np.array([0.0, 1.0, 0.0])  # Grün für Ziele
                    elif entity_type == 'obstacle':
                        return True, np.array([1.0, 0.0, 0.0])  # Rot für Hindernisse
                    elif entity_type == 'agent':
                        return True, np.array([0.0, 0.0, 1.0])  # Blau für andere Agenten
                    else:
                        return True, np.array([0.7, 0.7, 0.7])  # Grau für unbekannte Objekte
        except:
            pass  # Bei Fehlern in der Entitätsprüfung fortfahren
        
        # Keine Kollision gefunden, gib die Himmelsfarbe zurück
        return False, sky_color
    
    def _lidar_raycast(self, origin, direction, max_distance, environment):
        """
        Führt einen vereinfachten Raycast für LiDAR-Messungen durch.
        """
        # Prüfe Terrain-Kollision
        try:
            if environment.terrain is not None:
                hit_point, hit_distance = self._raycast_terrain(origin, direction, max_distance, environment.terrain)
                
                if hit_point is not None:
                    return True, hit_distance
        except:
            pass  # Bei Fehlern in der Terrain-Prüfung fortfahren
        
        # Prüfe Kollision mit Entitäten
        min_distance = max_distance
        hit_detected = False
        
        try:
            for entity in environment.get_visible_entities(origin, direction, max_distance):
                hit, distance = self._raycast_sphere(origin, direction, entity['position'], entity.get('collision_radius', 0.5))
                
                if hit and distance < min_distance:
                    min_distance = distance
                    hit_detected = True
        except:
            pass  # Bei Fehlern in der Entitätsprüfung fortfahren
        
        return hit_detected, min_distance
    
    def _raycast_sphere(self, ray_origin, ray_direction, sphere_center, sphere_radius):
        """
        Prüft, ob ein Strahl eine Kugel trifft.
        Gibt True zurück, wenn der Strahl die Kugel trifft, zusammen mit der Entfernung.
        """
        # Vektor von Strahlursprung zum Kugelmittelpunkt
        oc = ray_origin - sphere_center
        
        # Quadratische Gleichung für Schnittpunkte: t^2*dot(d,d) + 2*t*dot(oc,d) + dot(oc,oc) - r^2 = 0
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - sphere_radius * sphere_radius
        
        # Diskriminante
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            # Keine Schnittpunkte
            return False, float('inf')
        
        # Finde den nächsten Schnittpunkt
        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
        
        if t < 0:
            # Kugel ist hinter dem Strahlursprung
            return False, float('inf')
        
        return True, t
    
    def _raycast_terrain(self, ray_origin, ray_direction, max_distance, terrain):
        """
        Führt einen Raycast mit dem Terrain durch.
        """
        # Schrittgröße für den Raycast
        step_size = 0.1
        
        # Aktuelle Position entlang des Strahls
        current_pos = ray_origin.copy()
        
        # Gesamtdistanz entlang des Strahls
        total_distance = 0.0
        
        while total_distance < max_distance:
            # Prüfe, ob wir unter dem Terrain sind
            try:
                terrain_height = terrain.get_height(current_pos[0], current_pos[1])
                
                if current_pos[2] <= terrain_height:
                    # Strahl hat das Terrain getroffen
                    return current_pos, total_distance
            except:
                # Bei Fehlern in der Höhenabfrage fortfahren
                pass
            
            # Bewege entlang des Strahls
            current_pos += ray_direction * step_size
            total_distance += step_size
        
        # Keine Kollision innerhalb der maximalen Distanz
        return None, None
    
    def get_observation(self):
        """
        Gibt die aktuellen Sensordaten als Wörterbuch zurück.
        """
        return {sensor_type: data for sensor_type, data in self.last_sensor_data.items() if data is not None}


# Umfangreiche Umgebung mit dynamischen Elementen und Multiagent-Unterstützung
class Environment:
    """
    Umfassende Simulationsumgebung mit Terrain, Physik und dynamischen Objekten.
    """
    def __init__(self, config=None):
        # Standardkonfiguration
        self.default_config = {
            'size': (100, 100),
            'physics_detail_level': 2,
            'max_entities': 100,
            'entity_spawn_rate': 0.05,  # Pro Sekunde
            'day_night_cycle': True,
            'weather_effects': True,
            'wind_enabled': True
        }
        
        # Verwende benutzerdefinierte Konfiguration, falls angegeben
        self.config = self.default_config.copy()
        if config is not None:
            self.config.update(config)
        
        # Initialisiere Terrain und Physik
        self.terrain = Terrain(
            size=self.config['size'],
            resolution=1.0,
            detail_level=self.config['physics_detail_level']
        )
        
        self.physics = PhysicsEngine(detail_level=self.config['physics_detail_level'])
        
        # Zeit und Umgebungsbedingungen
        self.time = 0.0  # Simulationszeit in Sekunden
        self.day_time = 0.5  # 0.0=Mitternacht, 0.5=Mittag, 1.0=Mitternacht
        self.weather_condition = 'clear'  # 'clear', 'cloudy', 'rain', 'fog', 'snow'
        self.wind_direction = np.array([1.0, 0.0, 0.0])  # Normalisierter Vektor
        self.wind_speed = 2.0  # m/s
        
        # Entitäten in der Umgebung
        self.entities = []
        self.agents = []
        self.smell_sources = []
        
        # Entitäten-ID-Zähler
        self.next_entity_id = 0
        
        # Zeit seit dem letzten Spawnen einer Entität
        self.time_since_last_spawn = 0.0
        
        # Initialisiere die Umgebung
        self._initialize_environment()
    
    def _initialize_environment(self):
        """
        Initialisiert die Umgebung mit Startentitäten und Bedingungen.
        """
        # Erzeuge Geruchsquellen in der Umgebung
        self._create_smell_sources()
        
        # Initialisiere Wetterbedingungen
        self._update_weather()
        
        # Erzeuge initiale Entitäten
        num_initial_entities = self.config['max_entities'] // 5
        
        for _ in range(num_initial_entities):
            entity_type = np.random.choice(['obstacle', 'target', 'dynamic'])
            self._spawn_entity(entity_type)
    
    def _create_smell_sources(self):
        """
        Erzeugt Geruchsquellen in der Umgebung.
        """
        # Einige Zielgerüche
        num_targets = 3
        for i in range(num_targets):
            pos_x = np.random.uniform(-self.config['size'][0]/2, self.config['size'][0]/2)
            pos_y = np.random.uniform(-self.config['size'][1]/2, self.config['size'][1]/2)
            pos_z = self.terrain.get_height(pos_x, pos_y) + 0.5  # Leicht über dem Boden
            
            self.smell_sources.append({
                'id': len(self.smell_sources),
                'position': np.array([pos_x, pos_y, pos_z]),
                'intensity': np.random.uniform(1.0, 5.0),
                'type': 0  # 0 = Zielgeruch
            })
        
        # Einige Hindernis-/Gefahrengerüche
        num_hazards = 5
        for i in range(num_hazards):
            pos_x = np.random.uniform(-self.config['size'][0]/2, self.config['size'][0]/2)
            pos_y = np.random.uniform(-self.config['size'][1]/2, self.config['size'][1]/2)
            pos_z = self.terrain.get_height(pos_x, pos_y) + 0.3
            
            self.smell_sources.append({
                'id': len(self.smell_sources),
                'position': np.array([pos_x, pos_y, pos_z]),
                'intensity': np.random.uniform(0.5, 3.0),
                'type': 1  # 1 = Hindernis-/Gefahrengeruch
            })
        
        # Einige Wegpunkt-/Orientierungsgerüche
        num_waypoints = 4
        for i in range(num_waypoints):
            angle = i * (2 * np.pi / num_waypoints)
            distance = self.config['size'][0] / 4
            
            pos_x = distance * np.cos(angle)
            pos_y = distance * np.sin(angle)
            pos_z = self.terrain.get_height(pos_x, pos_y) + 1.0
            
            self.smell_sources.append({
                'id': len(self.smell_sources),
                'position': np.array([pos_x, pos_y, pos_z]),
                'intensity': np.random.uniform(2.0, 4.0),
                'type': 2  # 2 = Wegpunkt-/Orientierungsgeruch
            })
    
    def _spawn_entity(self, entity_type):
        """
        Erzeugt eine neue Entität in der Umgebung.
        """
        if len(self.entities) >= self.config['max_entities']:
            return None
        
        # Zufällige Position innerhalb der Umgebung
        pos_x = np.random.uniform(-self.config['size'][0]/2, self.config['size'][0]/2)
        pos_y = np.random.uniform(-self.config['size'][1]/2, self.config['size'][1]/2)
        pos_z = self.terrain.get_height(pos_x, pos_y) + 0.5  # Leicht über dem Boden
        
        # Zufällige Orientierung
        rotation = np.random.uniform(0, 2*np.pi, 3)
        
        # Quaternion aus Euler-Winkeln
        qx = np.sin(rotation[0]/2) * np.cos(rotation[1]/2) * np.cos(rotation[2]/2) - np.cos(rotation[0]/2) * np.sin(rotation[1]/2) * np.sin(rotation[2]/2)
        qy = np.cos(rotation[0]/2) * np.sin(rotation[1]/2) * np.cos(rotation[2]/2) + np.sin(rotation[0]/2) * np.cos(rotation[1]/2) * np.sin(rotation[2]/2)
        qz = np.cos(rotation[0]/2) * np.cos(rotation[1]/2) * np.sin(rotation[2]/2) - np.sin(rotation[0]/2) * np.sin(rotation[1]/2) * np.cos(rotation[2]/2)
        qw = np.cos(rotation[0]/2) * np.cos(rotation[1]/2) * np.cos(rotation[2]/2) + np.sin(rotation[0]/2) * np.sin(rotation[1]/2) * np.sin(rotation[2]/2)
        
        # Eigenschaften basierend auf Entitätstyp
        if entity_type == 'obstacle':
            # Statisches Hindernis
            entity = {
                'id': self.next_entity_id,
                'type': 'obstacle',
                'position': np.array([pos_x, pos_y, pos_z]),
                'rotation': rotation,
                'rotation_quat': np.array([qw, qx, qy, qz]),
                'velocity': np.zeros(3),
                'angular_velocity': np.zeros(3),
                'mass': np.random.uniform(5.0, 50.0),
                'collision_radius': np.random.uniform(0.3, 1.5),
                'elasticity': 0.3,
                'friction': 0.7,
                'color': np.array([1.0, 0.0, 0.0]),  # Rot
                'is_static': True
            }
        elif entity_type == 'target':
            # Zielobjekt
            entity = {
                'id': self.next_entity_id,
                'type': 'target',
                'position': np.array([pos_x, pos_y, pos_z]),
                'rotation': rotation,
                'rotation_quat': np.array([qw, qx, qy, qz]),
                'velocity': np.zeros(3),
                'angular_velocity': np.zeros(3),
                'mass': np.random.uniform(1.0, 10.0),
                'collision_radius': np.random.uniform(0.2, 0.8),
                'elasticity': 0.5,
                'friction': 0.3,
                'color': np.array([0.0, 1.0, 0.0]),  # Grün
                'is_static': True,
                'reward': np.random.uniform(1.0, 10.0)
            }
        elif entity_type == 'dynamic':
            # Dynamisches Objekt mit eigener Bewegung
            entity = {
                'id': self.next_entity_id,
                'type': 'dynamic',
                'position': np.array([pos_x, pos_y, pos_z]),
                'rotation': rotation,
                'rotation_quat': np.array([qw, qx, qy, qz]),
                'velocity': np.random.uniform(-1.0, 1.0, 3),
                'angular_velocity': np.random.uniform(-0.5, 0.5, 3),
                'mass': np.random.uniform(2.0, 20.0),
                'collision_radius': np.random.uniform(0.2, 1.0),
                'elasticity': 0.4,
                'friction': 0.5,
                'color': np.array([0.0, 0.0, 1.0]),  # Blau
                'is_static': False,
                'behavior': np.random.choice(['random', 'patrol', 'attract', 'repel']),
                'behavior_params': {
                    'speed': np.random.uniform(0.5, 2.0),
                    'direction_change_prob': 0.05,
                    'target_id': None
                }
            }
        else:
            # Unbekannter Entitätstyp
            return None
        
        # Berechne Trägheitsmoment basierend auf Masse und Radius (für eine Kugel)
        moment_of_inertia = 2/5 * entity['mass'] * entity['collision_radius']**2
        entity['inertia'] = np.array([moment_of_inertia, moment_of_inertia, moment_of_inertia])
        
        # Einzigartige ID zuweisen und zählen
        self.next_entity_id += 1
        
        # Entität zur Umgebung hinzufügen
        self.entities.append(entity)
        
        return entity
    def step(self, dt=0.01):
        """
        Führt einen Zeitschritt in der Simulationsumgebung durch.
        """
        # Zeit aktualisieren
        self.time += dt
        self.time_since_last_spawn += dt
        
        # Tag-Nacht-Zyklus aktualisieren, falls aktiviert
        if self.config['day_night_cycle']:
            day_length = 300.0  # 5 Minuten pro Tag für schnellere Simulation
            self.day_time = (self.time % day_length) / day_length
        
        # Wetter und Wind aktualisieren, falls aktiviert
        if self.config['weather_effects'] and np.random.random() < 0.001:
            self._update_weather()
        
        if self.config['wind_enabled'] and np.random.random() < 0.01:
            self._update_wind()
        
        # Spawne neue Entitäten basierend auf Spawn-Rate
        if self.time_since_last_spawn > 1.0 / self.config['entity_spawn_rate']:
            entity_type = np.random.choice(['obstacle', 'target', 'dynamic'], p=[0.3, 0.2, 0.5])
            self._spawn_entity(entity_type)
            self.time_since_last_spawn = 0.0
        
        # Aktualisiere alle Entitäten
        for entity in self.entities:
            # Statische Entitäten nicht aktualisieren
            if entity.get('is_static', False):
                continue
            
            # Anwenden von KI-Verhalten für dynamische Entitäten
            if entity['type'] == 'dynamic':
                self._update_entity_behavior(entity, dt)
            
            # Physikalische Aktualisierung
            forces = np.zeros(3)
            torques = np.zeros(3)
            
            # Gravitationskraft (wenn nicht ignoriert)
            if not entity.get('ignore_gravity', False):
                forces += np.array([0, 0, -9.81]) * entity['mass']
            
            # Luftwiderstand
            if self.config['physics_detail_level'] >= 2:
                velocity_squared = entity['velocity'] * np.abs(entity['velocity'])
                drag_coefficient = entity.get('drag_coefficient', 0.5)
                cross_section = entity.get('cross_section', entity['collision_radius']**2 * np.pi)
                air_density = 1.225  # kg/m^3
                
                drag_force = -0.5 * air_density * drag_coefficient * cross_section * velocity_squared
                forces += drag_force
            
            # Anwenden von Wind auf die Entität
            if self.config['wind_enabled'] and self.config['physics_detail_level'] >= 3:
                # Wind hat mehr Einfluss auf leichtere Objekte
                wind_force = self.wind_direction * self.wind_speed * 0.1 * entity['collision_radius']**2
                wind_force = wind_force * (1.0 / entity['mass'])  # Weniger Einfluss auf schwere Objekte
                forces += wind_force
            
            # Physik-Engine, um Kräfte und Bewegung anzuwenden
            self.physics.apply_forces(entity, forces, torques, dt)
        
        # Kollisionserkennung und -auflösung
        collisions = self.physics.detect_collisions(self.entities, self.terrain)
        self.physics.resolve_collisions(self.entities, collisions, self.terrain)
        
        # Aktualisiere Agenten
        for agent in self.agents:
            agent.update(dt, self)
        
        # Entferne Entitäten, die zu weit unten sind (aus der Welt gefallen)
        min_height = -10.0
        self.entities = [e for e in self.entities if e['position'][2] > min_height]
    
    def _update_entity_behavior(self, entity, dt):
        """
        Aktualisiert das Verhalten dynamischer Entitäten.
        """
        behavior = entity.get('behavior', 'random')
        params = entity.get('behavior_params', {})
        speed = params.get('speed', 1.0)
        
        if behavior == 'random':
            # Zufällige Richtungsänderung mit einer bestimmten Wahrscheinlichkeit
            if np.random.random() < params.get('direction_change_prob', 0.05):
                # Neue zufällige Richtung wählen
                direction = np.random.uniform(-1, 1, 3)
                direction[2] = 0  # Keine vertikale Bewegung
                direction = direction / np.linalg.norm(direction)
                
                # Geschwindigkeit in die neue Richtung setzen
                entity['velocity'] = direction * speed
        
        elif behavior == 'patrol':
            # Patrouilliere zwischen Wegpunkten
            if 'waypoints' not in params:
                # Erstelle Wegpunkte, wenn noch keine vorhanden sind
                num_waypoints = np.random.randint(2, 5)
                waypoints = []
                
                for _ in range(num_waypoints):
                    wp_x = np.random.uniform(-self.config['size'][0]/3, self.config['size'][0]/3)
                    wp_y = np.random.uniform(-self.config['size'][1]/3, self.config['size'][1]/3)
                    wp_z = self.terrain.get_height(wp_x, wp_y) + 0.5
                    waypoints.append(np.array([wp_x, wp_y, wp_z]))
                
                params['waypoints'] = waypoints
                params['current_waypoint'] = 0
            
            # Bewege zum aktuellen Wegpunkt
            current_wp_idx = params.get('current_waypoint', 0)
            current_wp = params['waypoints'][current_wp_idx]
            
            # Vektor zum Wegpunkt
            to_waypoint = current_wp - entity['position']
            distance = np.linalg.norm(to_waypoint[:2])
            
            if distance < 1.0:
                # Wegpunkt erreicht, gehe zum nächsten
                params['current_waypoint'] = (current_wp_idx + 1) % len(params['waypoints'])
            else:
                # Bewege in Richtung des Wegpunkts
                direction = to_waypoint / distance
                entity['velocity'] = direction * speed
        
        elif behavior == 'attract' or behavior == 'repel':
            # Versuche, sich zu einem Ziel hinzubewegen oder davon weg
            target_id = params.get('target_id')
            
            # Wähle ein neues Ziel, wenn keins vorhanden
            if target_id is None:
                # Suche nach einem geeigneten Ziel
                potential_targets = [e for e in self.entities if e['type'] == 'target' or 
                                     (e['type'] == 'dynamic' and e['id'] != entity['id'])]
                
                if potential_targets:
                    target = np.random.choice(potential_targets)
                    params['target_id'] = target['id']
                    target_id = target['id']
            
            # Finde das Ziel in der Entitätenliste
            target = None
            for e in self.entities:
                if e['id'] == target_id:
                    target = e
                    break
            
            if target:
                # Vektor zum Ziel
                to_target = target['position'] - entity['position']
                to_target[2] = 0  # Ignoriere Höhenunterschiede für die Bewegungsrichtung
                distance = np.linalg.norm(to_target)
                
                if distance > 0.1:
                    direction = to_target / distance
                    
                    # Für "repel" die Richtung umkehren
                    if behavior == 'repel':
                        direction = -direction
                    
                    entity['velocity'] = direction * speed
            else:
                # Ziel nicht mehr vorhanden, setze auf zufälliges Verhalten zurück
                entity['behavior'] = 'random'
        
        # Aktualisiere die Verhaltensparameter
        entity['behavior_params'] = params
    
    def _update_weather(self):
        """
        Aktualisiert die Wetterbedingungen in der Umgebung.
        """
        if not self.config['weather_effects']:
            return
        
        # Wetterbedingungen wechseln zufällig
        conditions = ['clear', 'cloudy', 'rain', 'fog', 'snow']
        weights = [0.4, 0.3, 0.15, 0.1, 0.05]
        
        self.weather_condition = np.random.choice(conditions, p=weights)
        
        # Bei schlechtem Wetter reduzierte Sichtweite
        if self.weather_condition in ['rain', 'fog', 'snow']:
            # Bei Agenten mit Sensorsystem die Sensorparameter anpassen
            for agent in self.agents:
                if hasattr(agent, 'sensor_system'):
                    # Sichtweite reduzieren
                    if 'camera' in agent.sensor_system.config:
                        agent.sensor_system.config['camera']['range'] *= 0.7
                    
                    # LiDAR-Reichweite reduzieren
                    if 'lidar' in agent.sensor_system.config:
                        agent.sensor_system.config['lidar']['range'] *= 0.8
                    
                    # Mehr Rauschen in Sensoren
                    for sensor in agent.sensor_system.config:
                        if 'noise_level' in agent.sensor_system.config[sensor]:
                            agent.sensor_system.config[sensor]['noise_level'] *= 1.5
    
    def _update_wind(self):
        """
        Aktualisiert die Windrichtung und -geschwindigkeit.
        """
        if not self.config['wind_enabled']:
            return
        
        # Zufällige leichte Änderung der Windrichtung
        angle_change = np.random.normal(0, 0.1)
        current_angle = np.arctan2(self.wind_direction[1], self.wind_direction[0])
        new_angle = current_angle + angle_change
        
        self.wind_direction[0] = np.cos(new_angle)
        self.wind_direction[1] = np.sin(new_angle)
        
        # Normalisieren
        self.wind_direction = self.wind_direction / np.linalg.norm(self.wind_direction)
        
        # Aktualisiere Windgeschwindigkeit
        self.wind_speed += np.random.normal(0, 0.2)
        self.wind_speed = max(0.5, min(10.0, self.wind_speed))  # Begrenze zwischen 0.5 und 10 m/s
    
    def get_entities_near(self, position, radius):
        """
        Gibt alle Entitäten in einem bestimmten Radius um eine Position zurück.
        """
        nearby_entities = []
        
        for entity in self.entities:
            distance = np.linalg.norm(entity['position'] - position)
            if distance <= radius + entity.get('collision_radius', 0.5):
                nearby_entities.append(entity)
        
        return nearby_entities
    
    def get_visible_entities(self, observer_position, direction, max_distance):
        """
        Gibt Entitäten zurück, die vom Beobachter aus sichtbar sind.
        """
        # Normalisiere die Richtung
        direction = direction / np.linalg.norm(direction)
        
        visible_entities = []
        
        for entity in self.entities:
            # Vektor von Beobachter zur Entität
            to_entity = entity['position'] - observer_position
            distance = np.linalg.norm(to_entity)
            
            # Zu weit entfernt
            if distance > max_distance:
                continue
            
            # Normalisiere den Vektor zur Entität
            if distance > 0:
                to_entity_normalized = to_entity / distance
            else:
                continue  # Ignore entities at exactly the same position
            
            # Skalarprodukt gibt den Kosinus des Winkels
            cosine = np.dot(direction, to_entity_normalized)
            
            # Sichtfeld-Prüfung: Wenn der Winkel klein genug ist (cosine nahe 1)
            if cosine > 0.7:  # Etwa 45 Grad Sichtfeld
                visible_entities.append(entity)
        
        return visible_entities
    
    def get_smell_sources(self):
        """
        Gibt alle Geruchsquellen in der Umgebung zurück.
        """
        return self.smell_sources
    
    def get_wind_direction(self):
        """
        Gibt die aktuelle Windrichtung zurück.
        """
        if self.config['wind_enabled']:
            return self.wind_direction
        else:
            return None
    
    def render(self, mode='human'):
        """
        Rendert die Umgebung.
        """
        if mode == 'human':
            # 3D-Darstellung der Umgebung
            fig = plt.figure(figsize=(15, 10))
            
            # Terrain in 3D darstellen
            ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
            self.terrain.render_3d(ax_3d, resolution_factor=1)
            
            # Entitäten im 3D-Plot hinzufügen
            for entity in self.entities:
                pos = entity['position']
                radius = entity.get('collision_radius', 0.5)
                color = entity.get('color', np.array([0.5, 0.5, 0.5]))
                
                # Kugeldarstellung
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = pos[0] + radius * np.cos(u) * np.sin(v)
                y = pos[1] + radius * np.sin(u) * np.sin(v)
                z = pos[2] + radius * np.cos(v)
                ax_3d.plot_surface(x, y, z, color=color, alpha=0.7)
                
                # Geschwindigkeitsvektor anzeigen, wenn nicht statisch
                if not entity.get('is_static', False) and np.linalg.norm(entity.get('velocity', np.zeros(3))) > 0.1:
                    velocity = entity['velocity']
                    velocity_norm = velocity / np.linalg.norm(velocity)
                    ax_3d.quiver(pos[0], pos[1], pos[2], 
                                velocity_norm[0], velocity_norm[1], velocity_norm[2], 
                                length=radius*2, color='black', arrow_length_ratio=0.3)
            
            # Agenten im 3D-Plot hinzufügen
            for agent in self.agents:
                pos = agent.get_position()
                ax_3d.scatter(pos[0], pos[1], pos[2], color='cyan', s=100, marker='^')
            
            # Geruchsquellen als transparente Kugeln darstellen
            for source in self.smell_sources:
                pos = source['position']
                intensity = source['intensity']
                source_type = source['type']
                
                # Farbe je nach Typ
                if source_type == 0:
                    color = [0, 1, 0, 0.3]  # Grün für Ziele
                elif source_type == 1:
                    color = [1, 0, 0, 0.3]  # Rot für Hindernisse
                else:
                    color = [0, 0, 1, 0.3]  # Blau für andere
                
                # Geruchsquelle als transparente Kugel mit Größe proportional zur Intensität
                u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:5j]
                x = pos[0] + 0.2 * np.cos(u) * np.sin(v)
                y = pos[1] + 0.2 * np.sin(u) * np.sin(v)
                z = pos[2] + 0.2 * np.cos(v)
                ax_3d.plot_surface(x, y, z, color=color, alpha=0.5)
                
                # Geruchsausbreitung als transparente Kugel
                u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:5j]
                x = pos[0] + intensity * np.cos(u) * np.sin(v)
                y = pos[1] + intensity * np.sin(u) * np.sin(v)
                z = pos[2] + intensity * np.cos(v)
                ax_3d.plot_surface(x, y, z, color=color, alpha=0.1)
            
            # 2D-Draufsicht der Umgebung
            ax_2d = fig.add_subplot(2, 2, 2)
            self.terrain.render_2d(ax_2d)
            
            # Entitäten in 2D darstellen
            for entity in self.entities:
                pos = entity['position']
                radius = entity.get('collision_radius', 0.5)
                color = entity.get('color', np.array([0.5, 0.5, 0.5]))
                
                # Kreis für die Entität
                circle = plt.Circle((pos[0], pos[1]), radius, color=color, alpha=0.7)
                ax_2d.add_patch(circle)
                
                # Geschwindigkeitsvektor anzeigen
                if not entity.get('is_static', False) and 'velocity' in entity:
                    vel = entity['velocity']
                    if np.linalg.norm(vel[:2]) > 0.1:  # Nur horizontale Komponente
                        ax_2d.arrow(pos[0], pos[1], vel[0], vel[1], 
                                   head_width=0.3, head_length=0.5, fc='black', ec='black')
            
            # Agenten in 2D darstellen
            for agent in self.agents:
                pos = agent.get_position()
                ax_2d.scatter(pos[0], pos[1], color='cyan', s=100, marker='^')
            
            # Umgebungsinformationen anzeigen
            ax_info = fig.add_subplot(2, 2, 3)
            ax_info.axis('off')
            
            info_text = f"Simulationszeit: {self.time:.2f}s\n"
            info_text += f"Tageszeit: {self.day_time:.2f}\n"
            info_text += f"Wetter: {self.weather_condition}\n"
            info_text += f"Windgeschwindigkeit: {self.wind_speed:.2f} m/s\n"
            info_text += f"Anzahl Entitäten: {len(self.entities)}\n"
            info_text += f"Anzahl Agenten: {len(self.agents)}\n"
            
            y_pos = 0.9
            for line in info_text.split('\n'):
                ax_info.text(0.1, y_pos, line, fontsize=12)
                y_pos -= 0.1
            
            # Wind-Visualisierung
            if self.config['wind_enabled']:
                ax_wind = fig.add_subplot(2, 2, 4)
                ax_wind.set_xlim(-1.2, 1.2)
                ax_wind.set_ylim(-1.2, 1.2)
                ax_wind.set_aspect('equal')
                ax_wind.set_title('Windrichtung und -stärke')
                
                # Kompass zeichnen
                circle = plt.Circle((0, 0), 1.0, fill=False, color='black')
                ax_wind.add_patch(circle)
                
                # Richtungsmarkierungen
                ax_wind.text(0, 1.1, 'N', fontsize=12, ha='center')
                ax_wind.text(1.1, 0, 'O', fontsize=12, va='center')
                ax_wind.text(0, -1.1, 'S', fontsize=12, ha='center')
                ax_wind.text(-1.1, 0, 'W', fontsize=12, va='center')
                
                # Windpfeil zeichnen
                wind_dir = self.wind_direction[:2]  # Nur horizontale Komponente
                wind_length = min(1.0, self.wind_speed / 10.0)  # Skaliere die Länge
                
                ax_wind.arrow(0, 0, wind_dir[0] * wind_length, wind_dir[1] * wind_length, 
                             head_width=0.1, head_length=0.2, fc='blue', ec='blue', width=0.05)
            
            plt.tight_layout()
            plt.show()
            
        elif mode == 'rgb_array':
            # RGB-Array für maschinelles Lernen zurückgeben
            # Für dieses Beispiel vereinfacht implementiert
            img_size = 500
            rgb_array = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            
            # Himmel zeichnen
            if self.weather_condition == 'clear':
                sky_color = [135, 206, 235]  # Hellblau
            elif self.weather_condition == 'cloudy':
                sky_color = [180, 180, 180]  # Grau
            elif self.weather_condition == 'rain':
                sky_color = [100, 100, 100]  # Dunkelgrau
            elif self.weather_condition == 'fog':
                sky_color = [200, 200, 200]  # Hellgrau
            elif self.weather_condition == 'snow':
                sky_color = [240, 240, 250]  # Fast weiß
            
            for y in range(img_size // 2):
                for x in range(img_size):
                    # Farbverlauf vom Horizont zum Himmel
                    interp = y / (img_size // 2)
                    r = int((1 - interp) * 255 + interp * sky_color[0])
                    g = int((1 - interp) * 255 + interp * sky_color[1])
                    b = int((1 - interp) * 255 + interp * sky_color[2])
                    rgb_array[y, x] = [r, g, b]
            
            # Boden zeichnen (sehr vereinfacht)
            for y in range(img_size // 2, img_size):
                for x in range(img_size):
                    # Höhe an dieser Position abfragen
                    rel_x = (x / img_size - 0.5) * self.config['size'][0]
                    rel_y = ((y - img_size // 2) / (img_size // 2)) * self.config['size'][1] / 2
                    
                    try:
                        height = self.terrain.get_height(rel_x, rel_y)
                        material = self.terrain.get_material(rel_x, rel_y)
                        
                        # Farbzuordnung basierend auf Material
                        if material == 'concrete':
                            color = [128, 128, 128]  # Grau
                        elif material == 'grass':
                            color = [34, 139, 34]    # Grün
                        elif material == 'sand':
                            color = [210, 180, 140]  # Sandfarben
                        elif material == 'ice':
                            color = [200, 200, 255]  # Hellblau
                        elif material == 'metal':
                            color = [169, 169, 169]  # Silber
                        elif material == 'rubber':
                            color = [50, 50, 50]     # Dunkelgrau
                        elif material == 'wood':
                            color = [139, 69, 19]    # Braun
                        else:
                            color = [100, 100, 100]  # Fallback
                        
                        # Höheninformation in die Farbe einbeziehen
                        height_factor = max(0, min(1, (height + 2) / 10))
                        r = int(color[0] * height_factor)
                        g = int(color[1] * height_factor)
                        b = int(color[2] * height_factor)
                        
                        rgb_array[y, x] = [r, g, b]
                    except:
                        # Bei Fehler grauen Pixel setzen
                        rgb_array[y, x] = [100, 100, 100]
            
            # Entitäten zeichnen (sehr vereinfacht)
            for entity in self.entities:
                try:
                    # Position in Bildkoordinaten umrechnen
                    screen_x = int((entity['position'][0] / self.config['size'][0] + 0.5) * img_size)
                    screen_y = int(img_size // 2 + entity['position'][1] / self.config['size'][1] * img_size // 2)
                    
                    # Farbe der Entität
                    color = entity.get('color', np.array([0.5, 0.5, 0.5]))
                    r, g, b = int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                    
                    # Größe basierend auf Kollisionsradius
                    radius = int(entity.get('collision_radius', 0.5) * 10)
                    
                    # Kreis zeichnen (sehr rudimentär)
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            if dx*dx + dy*dy <= radius*radius:
                                x = screen_x + dx
                                y = screen_y + dy
                                if 0 <= x < img_size and 0 <= y < img_size:
                                    rgb_array[y, x] = [r, g, b]
                except:
                    continue
            
            # Agenten zeichnen
            for agent in self.agents:
                try:
                    pos = agent.get_position()
                    screen_x = int((pos[0] / self.config['size'][0] + 0.5) * img_size)
                    screen_y = int(img_size // 2 + pos[1] / self.config['size'][1] * img_size // 2)
                    
                    # Agenten als cyan-farbenes Dreieck
                    agent_size = 8
                    for dy in range(-agent_size, agent_size + 1):
                        for dx in range(-agent_size, agent_size + 1):
                            if dy > -dx/2 - agent_size/2 and dy > dx/2 - agent_size/2 and dy < agent_size:
                                x = screen_x + dx
                                y = screen_y + dy
                                if 0 <= x < img_size and 0 <= y < img_size:
                                    rgb_array[y, x] = [0, 255, 255]  # Cyan
                except:
                    continue
            
            return rgb_array


# Roboterklasse mit fortschrittlicher Sensorik und Physik
class Robot:
    """
    Ein simulierter Roboter mit realistischer Physik, Sensorik und KI-Steuerung.
    """
    def __init__(self, config=None):
        # Standardkonfiguration
        self.default_config = {
            'mass': 15.0,
            'dimension': np.array([0.5, 0.5, 0.3]),  # Länge, Breite, Höhe in Metern
            'max_speed': 2.0,  # m/s
            'max_angular_speed': 1.0,  # rad/s
            'max_acceleration': 1.0,  # m/s²
            'max_angular_acceleration': 0.5,  # rad/s²
            'wheel_friction': 0.7,
            'battery_capacity': 1000.0,  # Wh
            'power_consumption': 50.0,  # W
            'sensor_detail_level': 2  # 1 (niedrig) bis 4 (ultra)
        }
        
        # Verwende benutzerdefinierte Konfiguration, falls angegeben
        self.config = self.default_config.copy()
        if config is not None:
            self.config.update(config)
        
        # Physikalischer Zustand
        self.position = np.zeros(3)
        self.rotation = np.zeros(3)  # Roll, Pitch, Yaw
        self.rotation_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Einheitsquaternion
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        
        # Robotermodell für Kollisionen
        self.collision_radius = max(self.config['dimension'][0], self.config['dimension'][1]) / 2
        
        # Interne Zustände
        self.battery_level = self.config['battery_capacity']
        self.motor_states = np.zeros(4)  # 4 Räder oder Motoren
        self.internal_temperature = 20.0  # Grad Celsius
        
        # Sensor-Setup
        self.sensor_config = {
            'camera': {
                'resolution': (64, 64) if self.config['sensor_detail_level'] <= 2 else (128, 128),
                'fov': 90,
                'range': 20.0,
                'noise_level': 0.02,
                'update_rate': 30
            },
            'lidar': {
                'num_rays': 16 if self.config['sensor_detail_level'] <= 2 else 32,
                'fov': 360,
                'range': 30.0,
                'noise_level': 0.01,
                'update_rate': 20
            },
            'imu': {
                'noise_level': 0.005,
                'drift_level': 0.001,
                'update_rate': 100
            },
            'touch': {
                'num_sensors': 8,
                'sensitivity': 0.8,
                'noise_level': 0.01,
                'update_rate': 50
            },
            'smell': {
                'num_sensors': 8,
                'sensitivity': 0.7,
                'range': 15.0,
                'noise_level': 0.05,
                'update_rate': 10
            },
            'gps': {
                'noise_level': 0.5,
                'update_rate': 5
            }
        }
        
        # Sensorsystem initialisieren
        self.sensor_system = SensorSystem(self.sensor_config)
        
        # Kontrollvariablen
        self.target_velocity = np.zeros(3)
        self.target_angular_velocity = np.zeros(3)
        
        # KI-Steuerung
        self.controller = None
    
    def set_position(self, position, rotation=None):
        """
        Setzt die Position und optional die Rotation des Roboters.
        """
        self.position = np.array(position)
        
        if rotation is not None:
            self.rotation = np.array(rotation)
            
            # Quaternion aus Euler-Winkeln berechnen
            roll, pitch, yaw = self.rotation
            qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
            qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
            qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
            qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
            
            self.rotation_quat = np.array([qw, qx, qy, qz])
    
    def get_position(self):
        """
        Gibt die aktuelle Position des Roboters zurück.
        """
        return self.position
    
    def get_rotation(self):
        """
        Gibt die aktuelle Rotation des Roboters zurück.
        """
        return self.rotation
    
    def set_controller(self, controller):
        """
        Setzt den Controller zur Steuerung des Roboters.
        """
        self.controller = controller
    
    def set_target_velocity(self, linear_velocity, angular_velocity=None):
        """
        Setzt die Zielgeschwindigkeit für den Roboter.
        
        linear_velocity: Linearer Geschwindigkeitsvektor [vx, vy, vz]
        angular_velocity: Winkelgeschwindigkeitsvektor [wx, wy, wz]
        """
        # Begrenze die lineare Geschwindigkeit
        speed = np.linalg.norm(linear_velocity)
        if speed > self.config['max_speed']:
            linear_velocity = linear_velocity * (self.config['max_speed'] / speed)
        
        self.target_velocity = np.array(linear_velocity)
        
        # Setze auch die Winkelgeschwindigkeit, falls angegeben
        if angular_velocity is not None:
            # Begrenze die Winkelgeschwindigkeit
            angular_speed = np.linalg.norm(angular_velocity)
            if angular_speed > self.config['max_angular_speed']:
                angular_velocity = angular_velocity * (self.config['max_angular_speed'] / angular_speed)
            
            self.target_angular_velocity = np.array(angular_velocity)
    
    def update(self, dt, environment):
        """
        Aktualisiert den Roboterzustand für einen Zeitschritt.
        
        dt: Zeitschrittgröße in Sekunden
        environment: Umgebungsobjekt für Sensor- und Physikinteraktionen
        """
        # Aktualisiere Sensordaten
        self.sensor_system.update(environment.time, self._get_state_dict(), environment)
        
        # Rufe den Controller auf, um Steuerungsentscheidungen zu treffen
        if self.controller is not None:
            observations = self.sensor_system.get_observation()
            controls = self.controller.compute_control(observations)
            
            # Setze Zielgeschwindigkeiten basierend auf Controller-Ausgabe
            if 'linear_velocity' in controls:
                self.target_velocity = controls['linear_velocity']
            
            if 'angular_velocity' in controls:
                self.target_angular_velocity = controls['angular_velocity']
        
        # Physikalische Aktualisierung
        self._update_physics(dt, environment)
        
        # Batterieverbrauch simulieren
        power_draw = self.config['power_consumption']
        
        # Zusätzlicher Verbrauch bei Bewegung
        velocity_factor = np.linalg.norm(self.velocity) / self.config['max_speed']
        angular_velocity_factor = np.linalg.norm(self.angular_velocity) / self.config['max_angular_speed']
        
        power_draw += 20.0 * velocity_factor + 10.0 * angular_velocity_factor
        
        # Batterie entladen (Wh zu Ws umrechnen mit * 3600)
        energy_used = power_draw * dt / 3600.0
        self.battery_level = max(0.0, self.battery_level - energy_used)
    
    def _update_physics(self, dt, environment):
        """
        Aktualisiert die physikalischen Eigenschaften des Roboters.
        """
        # Berechne Beschleunigung basierend auf Zielgeschwindigkeit
        acceleration = (self.target_velocity - self.velocity) / dt
        
        # Begrenze die Beschleunigung
        accel_magnitude = np.linalg.norm(acceleration)
        if accel_magnitude > self.config['max_acceleration']:
            acceleration = acceleration * (self.config['max_acceleration'] / accel_magnitude)
        
        # Berechne Winkelbeschleunigung
        angular_acceleration = (self.target_angular_velocity - self.angular_velocity) / dt
        
        # Begrenze die Winkelbeschleunigung
        angular_accel_magnitude = np.linalg.norm(angular_acceleration)
        if angular_accel_magnitude > self.config['max_angular_acceleration']:
            angular_acceleration = angular_acceleration * (self.config['max_angular_acceleration'] / angular_accel_magnitude)
        
        # Aktualisiere Geschwindigkeit
        self.velocity += acceleration * dt
        
        # Aktualisiere Winkelgeschwindigkeit
        self.angular_velocity += angular_acceleration * dt
        
        # Berechne Kräfte und Drehmomente
        forces = self.config['mass'] * acceleration
        
        # Berechne Trägheitsmoment (vereinfacht für einen Quader)
        w, d, h = self.config['dimension']
        Ixx = (1/12) * self.config['mass'] * (d**2 + h**2)
        Iyy = (1/12) * self.config['mass'] * (w**2 + h**2)
        Izz = (1/12) * self.config['mass'] * (w**2 + d**2)
        inertia = np.array([Ixx, Iyy, Izz])
        
        torques = inertia * angular_acceleration
        
        # Zustandswörterbuch für die Physik-Engine
        state = self._get_state_dict()
        
        # Anwenden von Physik
        updated_state = environment.physics.apply_forces(state, forces, torques, dt)
        
        # Aktualisiere Roboterzustand aus dem aktualisierten Zustandswörterbuch
        self.position = updated_state['position']
        self.rotation = updated_state['rotation']
        self.rotation_quat = updated_state['rotation_quat']
        self.velocity = updated_state['velocity']
        self.angular_velocity = updated_state['angular_velocity']
        
        # Zusätzliche Physikeffekte
        
        # Reibung mit dem Boden
        if environment.terrain is not None:
            terrain_height = environment.terrain.get_height(self.position[0], self.position[1])
            
            if self.position[2] <= terrain_height + 0.1:
                # Roboter ist am Boden, wende Reibung an
                material = environment.terrain.get_material(self.position[0], self.position[1])
                
                # Materialreibung abrufen
                friction_coefficient = environment.physics.friction_coefficients.get(material, 0.5)
                
                # Reibungskraft berechnen
                friction_magnitude = friction_coefficient * self.config['mass'] * 9.81
                
                # Horizontale Geschwindigkeitskomponente
                horizontal_velocity = self.velocity.copy()
                horizontal_velocity[2] = 0
                speed = np.linalg.norm(horizontal_velocity)
                
                if speed > 0.01:
                    # Richtung der Reibungskraft
                    friction_direction = -horizontal_velocity / speed
                    
                    # Reibungskraft anwenden
                    friction_force = friction_direction * min(friction_magnitude, self.config['mass'] * speed / dt)
                    
                    # Geschwindigkeit aktualisieren
                    self.velocity[:2] += friction_force[:2] * dt / self.config['mass']
    
    def _get_state_dict(self):
        """
        Gibt den aktuellen Roboterzustand als Dictionary für die Physik-Engine zurück.
        """
        return {
            'id': -1,  # Spezielle ID für den Roboter
            'type': 'robot',
            'position': self.position,
            'rotation': self.rotation,
            'rotation_quat': self.rotation_quat,
            'velocity': self.velocity,
            'angular_velocity': self.angular_velocity,
            'mass': self.config['mass'],
            'inertia': np.array([0.1, 0.1, 0.1]) * self.config['mass'],  # Vereinfacht
            'collision_radius': self.collision_radius,
            'elasticity': 0.3,
            'friction': self.config['wheel_friction'],
            'color': np.array([0.0, 0.5, 1.0]),  # Blau
            'is_static': False
        }
    
    def get_observation(self):
        """
        Gibt die aktuellen Sensordaten zurück.
        """
        return self.sensor_system.get_observation()


# Neuronales Netzwerk-basierter Controller für den Roboter
class RobotNeuralController:
    """
    Controller, der ein neuronales Netzwerk verwendet, um Sensordaten in Steuerungsanweisungen umzuwandeln.
    """
    def __init__(self, input_shapes, output_dim=5, hidden_layers=(128, 64)):
        """
        Initialisiert den Controller mit gegebenen Eingabe- und Ausgabedimensionen.
        
        input_shapes: Dictionary mit Sensorname als Schlüssel und Sensorform als Wert
        output_dim: Dimension des Ausgabevektors (typischerweise 5: vx, vy, vz, wx, wy, wz)
        hidden_layers: Tupel mit der Anzahl an Neuronen in den versteckten Schichten
        """
        self.input_shapes = input_shapes
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        
        # Baue das neuronale Netzwerk
        self.model = self._build_model()
        
        # Trainingshistorie
        self.training_history = []
    
    def _build_model(self):
        """
        Baut ein neuronales Netzwerk zur Verarbeitung der Sensordaten.
        """
        inputs = []
        processed_inputs = []
        
        # Verarbeite jeden Sensortyp mit spezialisierten Netzwerkeingängen
        for sensor_name, shape in self.input_shapes.items():
            if sensor_name == 'camera':
                # CNN für Kamerabilder
                img_input = Input(shape=shape)
                x = Conv2D(16, (3, 3), activation='relu', padding='same')(img_input)
                x = MaxPooling2D((2, 2))(x)
                x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
                x = MaxPooling2D((2, 2))(x)
                x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
                x = Flatten()(x)
                x = Dense(64, activation='relu')(x)
                
                inputs.append(img_input)
                processed_inputs.append(x)
            
            elif sensor_name == 'lidar':
                # 1D-Verarbeitung für LiDAR-Daten
                lidar_input = Input(shape=(shape,))
                x = Dense(32, activation='relu')(lidar_input)
                x = Dense(16, activation='relu')(x)
                
                inputs.append(lidar_input)
                processed_inputs.append(x)
            
            elif sensor_name == 'imu':
                # Verarbeitung für IMU-Daten (Beschleunigung und Winkelgeschwindigkeit)
                imu_input = Input(shape=(6,))  # 3D Beschleunigung + 3D Winkelgeschwindigkeit
                x = Dense(16, activation='relu')(imu_input)
                
                inputs.append(imu_input)
                processed_inputs.append(x)
            
            elif sensor_name == 'touch':
                # Berührungssensoren
                touch_input = Input(shape=(shape,))
                x = Dense(16, activation='relu')(touch_input)
                
                inputs.append(touch_input)
                processed_inputs.append(x)
            
            elif sensor_name == 'smell':
                # Geruchssensoren
                smell_input = Input(shape=(shape[0] * shape[1],))
                x = Dense(16, activation='relu')(smell_input)
                
                inputs.append(smell_input)
                processed_inputs.append(x)
            
            elif sensor_name == 'gps':
                # GPS-Daten
                gps_input = Input(shape=(2,))
                x = Dense(8, activation='relu')(gps_input)
                
                inputs.append(gps_input)
                processed_inputs.append(x)
        
        # Kombiniere alle verarbeiteten Eingaben
        if len(processed_inputs) > 1:
            combined = Concatenate()(processed_inputs)
        elif len(processed_inputs) == 1:
            combined = processed_inputs[0]
        else:
            # Fallback wenn keine Sensordaten verfügbar sind
            print("Warnung: Keine Sensordaten verfügbar. Erstelle Fallback-Eingabeschicht.")
            dummy_input = Input(shape=(10,))
            combined = Dense(32, activation='relu')(dummy_input)
            inputs.append(dummy_input)
        
        # Füge versteckte Schichten hinzu
        for units in self.hidden_layers:
            combined = Dense(units, activation='relu')(combined)
        
        # Ausgabeschicht für Steuerungsanweisungen
        # Verwende tanh für normalisierte Ausgaben zwischen -1 und 1
        outputs = Dense(self.output_dim, activation='tanh')(combined)
        
        # Erstelle das Modell
        model = Model(inputs=inputs, outputs=outputs)
        
        # Kompiliere das Modell
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model
    def compute_control(self, observations):
        """
        Berechnet Steuerungsbefehle basierend auf Sensordaten.
        
        observations: Dictionary mit Sensordaten
        Returns: Dictionary mit Steuerungsanweisungen
        """
        # Bereite die Eingabedaten für das Netzwerk vor
        inputs = []
        
        # Für jeden erwarteten Sensortyp
        for sensor_name, shape in self.input_shapes.items():
            if sensor_name in observations:
                sensor_data = observations[sensor_name]
                
                # Besondere Behandlung für verschiedene Sensortypen
                if sensor_name == 'camera':
                    # Normalisiere das Bild auf Werte zwischen 0 und 1
                    norm_data = sensor_data.astype(np.float32) / 255.0
                    inputs.append(np.expand_dims(norm_data, axis=0))
                
                elif sensor_name == 'smell':
                    # Flatten die Geruchsdaten
                    flat_data = sensor_data.flatten().astype(np.float32)
                    inputs.append(np.expand_dims(flat_data, axis=0))
                
                elif sensor_name == 'imu':
                    # Kombiniere Beschleunigung und Winkelgeschwindigkeit
                    imu_data = np.concatenate([
                        sensor_data['acceleration'],
                        sensor_data['angular_velocity']
                    ]).astype(np.float32)
                    inputs.append(np.expand_dims(imu_data, axis=0))
                
                else:
                    # Für andere Sensoren
                    inputs.append(np.expand_dims(sensor_data.astype(np.float32), axis=0))
            else:
                # Wenn dieser Sensor nicht in den Beobachtungen enthalten ist,
                # verwende Nullen der entsprechenden Form
                if sensor_name == 'imu':
                    # IMU hat spezielle Behandlung mit 6 Werten
                    inputs.append(np.zeros((1, 6), dtype=np.float32))
                else:
                    # Für andere Sensoren
                    inputs.append(np.zeros((1,) + tuple(shape), dtype=np.float32))
        
        # Vorhersage mit dem Modell machen
        control_output = self.model.predict(inputs, verbose=0)[0]
        
        # Umwandeln in ein Steuerungsdictionary
        controls = {
            'linear_velocity': control_output[:3],
            'angular_velocity': control_output[3:]
        }
        
        return controls
    
    def train(self, dataset, epochs=10, batch_size=32, validation_split=0.2):
        """
        Trainiert das neuronale Netzwerk mit dem gegebenen Dataset.
        
        dataset: Tupel aus (X, y), wobei X eine Liste von Sensordaten und y die Steuerungsziele sind
        """
        X, y = dataset
        
        # Trainiere das Modell
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        # Speichere die Trainingshistorie
        self.training_history.append(history.history)
        
        return history
    
    def save(self, filepath):
        """
        Speichert das Modell.
        """
        self.model.save(filepath)
    
    def load(self, filepath):
        """
        Lädt ein vortrainiertes Modell.
        """
        self.model = tf.keras.models.load_model(filepath)


# Funktionen zur Ausführung der Simulation
def create_robot_controller(robot, environment):
    """
    Erstellt einen neuronalen Controller für den Roboter.
    """
    # Hole Beobachtungsstrukturen vom Roboter
    observations = robot.get_observation()
    
    # Erstelle Eingabeformwörterbuch für den Controller
    input_shapes = {}
    
    for sensor_name, data in observations.items():
        if sensor_name == 'camera':
            input_shapes[sensor_name] = data.shape
        elif sensor_name == 'lidar':
            input_shapes[sensor_name] = data.shape[0]
        elif sensor_name == 'imu':
            # IMU-Daten haben spezielle Struktur
            input_shapes[sensor_name] = 6  # 3D Beschleunigung + 3D Winkelgeschwindigkeit
        elif sensor_name == 'touch':
            input_shapes[sensor_name] = data.shape[0]
        elif sensor_name == 'smell':
            input_shapes[sensor_name] = data.shape
        elif sensor_name == 'gps':
            input_shapes[sensor_name] = 2
    
    # Erstelle und gib den Controller zurück
    return RobotNeuralController(input_shapes)


def generate_training_data(robot, environment, num_steps=1000):
    """
    Generiert Trainingsdaten für den Controller durch Simulation.
    
    Verwendet eine einfache Regelbasierte Steuerung, um Beispieldaten zu erzeugen.
    """
    X_data = {sensor: [] for sensor in robot.get_observation().keys()}
    y_data = []
    
    # Setze den Roboter auf eine zufällige Position
    random_x = np.random.uniform(-environment.config['size'][0]/3, environment.config['size'][0]/3)
    random_y = np.random.uniform(-environment.config['size'][1]/3, environment.config['size'][1]/3)
    terrain_height = environment.terrain.get_height(random_x, random_y)
    robot.set_position([random_x, random_y, terrain_height + 0.5])
    
    # Führe die Simulation für eine bestimmte Anzahl von Schritten durch
    for step in range(num_steps):
        # Hole aktuelle Beobachtungen
        observations = robot.get_observation()
        
        # Implementiere eine einfache regelbasierte Steuerung, um Beispieldaten zu generieren
        # In einer realen Anwendung könnte dies von menschlichen Benutzern oder einer anderen Quelle kommen
        
        # Finde das nächste Ziel oder vermeide das nächste Hindernis
        target_pos = None
        obstacle_pos = None
        
        # Verwende LiDAR, um Objekte zu erkennen
        if 'lidar' in observations:
            lidar_data = observations['lidar']
            min_distance_idx = np.argmin(lidar_data)
            min_distance = lidar_data[min_distance_idx]
            
            if min_distance < 5.0:
                # Objekt in der Nähe, prüfe mit anderen Sensoren, ob es ein Ziel oder Hindernis ist
                angle = min_distance_idx * (2 * np.pi / len(lidar_data))
                direction = np.array([np.cos(angle), np.sin(angle), 0])
                
                # Vereinfachte Annahme: Wenn Geruchssensor stark ist, ist es ein Ziel, sonst ein Hindernis
                if 'smell' in observations and np.max(observations['smell'][:, 0]) > 0.5:
                    # Starker Geruch deutet auf ein Ziel hin
                    target_pos = robot.position + direction * min_distance
                else:
                    # Ansonsten als Hindernis betrachten
                    obstacle_pos = robot.position + direction * min_distance
        
        # Entscheide über die Steuerungsaktion
        linear_velocity = np.zeros(3)
        angular_velocity = np.zeros(3)
        
        if target_pos is not None:
            # Bewege zum Ziel
            direction_to_target = target_pos - robot.position
            distance = np.linalg.norm(direction_to_target[:2])
            
            if distance > 0.1:
                # Richtung normalisieren und lineare Geschwindigkeit setzen
                direction_to_target[:2] = direction_to_target[:2] / distance
                linear_velocity[:2] = direction_to_target[:2] * min(1.0, distance)
                
                # Berechne gewünschte Ausrichtung (Yaw)
                target_yaw = np.arctan2(direction_to_target[1], direction_to_target[0])
                current_yaw = robot.rotation[2]
                
                # Normalisierte Winkeldifferenz (-pi bis pi)
                angle_diff = np.arctan2(np.sin(target_yaw - current_yaw), np.cos(target_yaw - current_yaw))
                
                # Setze Winkelgeschwindigkeit, um sich zum Ziel zu drehen
                angular_velocity[2] = angle_diff * 0.5
        
        elif obstacle_pos is not None:
            # Vermeide Hindernis
            direction_to_obstacle = obstacle_pos - robot.position
            distance = np.linalg.norm(direction_to_obstacle[:2])
            
            if distance < 3.0:
                # Weiche aus, indem senkrecht zum Hindernis bewegt wird
                avoidance_direction = np.array([-direction_to_obstacle[1], direction_to_obstacle[0], 0])
                avoidance_direction = avoidance_direction / np.linalg.norm(avoidance_direction[:2])
                
                # Ausweichgeschwindigkeit proportional zur Nähe des Hindernisses
                avoidance_speed = max(0.2, 1.0 - distance / 3.0)
                linear_velocity = avoidance_direction * avoidance_speed
        
        else:
            # Keine Objekte in der Nähe, zufällige Erkundung
            if step % 50 == 0:  # Ändere Richtung alle 50 Schritte
                random_direction = np.random.uniform(-1, 1, 2)
                random_direction = random_direction / np.linalg.norm(random_direction)
                linear_velocity[:2] = random_direction * 0.5
                
                # Zufällige Drehung
                angular_velocity[2] = np.random.uniform(-0.3, 0.3)
        
        # Begrenzen Sie die Geschwindigkeiten
        linear_speed = np.linalg.norm(linear_velocity)
        if linear_speed > robot.config['max_speed']:
            linear_velocity = linear_velocity * (robot.config['max_speed'] / linear_speed)
        
        angular_speed = np.linalg.norm(angular_velocity)
        if angular_speed > robot.config['max_angular_speed']:
            angular_velocity = angular_velocity * (robot.config['max_angular_speed'] / angular_speed)
        
        # Setze die Zielgeschwindigkeit des Roboters
        robot.set_target_velocity(linear_velocity, angular_velocity)
        
        # Aktualisiere die Umgebung und den Roboter
        environment.step(0.1)
        robot.update(0.1, environment)
        
        # Speichere die Daten
        for sensor, data in observations.items():
            X_data[sensor].append(data.copy())
        
        # Kombiniere lineare und Winkelgeschwindigkeit für das Ausgabeziel
        control_output = np.concatenate([linear_velocity, angular_velocity])
        y_data.append(control_output)
    
    # Konvertiere Listen zu Arrays
    X_processed = []
    
    # Konvertiere die Sensordaten in das richtige Format für das Training
    for sensor in X_data.keys():
        if sensor == 'imu':
            # Kombiniere Beschleunigung und Winkelgeschwindigkeit
            imu_combined = []
            for data in X_data[sensor]:
                combined = np.concatenate([data['acceleration'], data['angular_velocity']])
                imu_combined.append(combined)
            X_processed.append(np.array(imu_combined))
        elif sensor == 'smell':
            # Flatten die Geruchsdaten
            smell_flat = [data.flatten() for data in X_data[sensor]]
            X_processed.append(np.array(smell_flat))
        else:
            X_processed.append(np.array(X_data[sensor]))
    
    y_processed = np.array(y_data)
    
    return X_processed, y_processed


def run_simulation(gpu_info=None):
    """
    Führt die vollständige Robotersimulation aus.
    """
    print(f"{TermColors.HEADER}Erweiterte Robotersimulation wird gestartet...{TermColors.ENDC}")
    
    # Hardware-Informationen abrufen
    if gpu_info is None:
        hardware_manager = HardwareManager()
        gpu_info = hardware_manager.gpu_info
        simulation_config = hardware_manager.get_simulation_config()
    else:
        simulation_config = {
            'resolution': 64,
            'max_entities': 100,
            'physics_detail': 2
        }
    
    print(f"{TermColors.OKBLUE}Simulation wird mit folgender Konfiguration ausgeführt:{TermColors.ENDC}")
    print(f"  Resolution: {simulation_config['resolution']}px")
    print(f"  Maximale Entitäten: {simulation_config['max_entities']}")
    print(f"  Physik-Detaillevel: {simulation_config['physics_detail']}")
    
    # Umgebung erstellen
    env_config = {
        'size': (100, 100),
        'physics_detail_level': simulation_config['physics_detail'],
        'max_entities': simulation_config['max_entities']
    }
    
    environment = Environment(env_config)
    
    # Roboter erstellen
    robot_config = {
        'mass': 15.0,
        'dimension': np.array([0.5, 0.5, 0.3]),
        'sensor_detail_level': simulation_config['physics_detail']
    }
    
    robot = Robot(robot_config)
    
    # Setze Roboter auf eine gute Startposition
    terrain_height = environment.terrain.get_height(0, 0)
    robot.set_position([0, 0, terrain_height + 0.5])
    
    # Registriere den Roboter in der Umgebung
    environment.agents.append(robot)
    
    # Controller erstellen
    print(f"{TermColors.OKBLUE}Erstelle neuronalen Controller für den Roboter...{TermColors.ENDC}")
    controller = create_robot_controller(robot, environment)
    
    # Trainingsdaten generieren
    print(f"{TermColors.OKBLUE}Generiere Trainingsdaten...{TermColors.ENDC}")
    X_train, y_train = generate_training_data(robot, environment, num_steps=500)
    
    # Controller trainieren
    print(f"{TermColors.OKBLUE}Trainiere neuronalen Controller...{TermColors.ENDC}")
    controller.train((X_train, y_train), epochs=5, batch_size=32)
    
    # Controller speichern
    controller.save("robot_controller.h5")
    
    # Controller dem Roboter zuweisen
    robot.set_controller(controller)
    
    # Testlauf durchführen
    print(f"{TermColors.OKGREEN}Starte Testlauf mit dem trainierten Controller...{TermColors.ENDC}")
    
    # Setze den Roboter an eine neue Position
    random_x = np.random.uniform(-environment.config['size'][0]/3, environment.config['size'][0]/3)
    random_y = np.random.uniform(-environment.config['size'][1]/3, environment.config['size'][1]/3)
    terrain_height = environment.terrain.get_height(random_x, random_y)
    robot.set_position([random_x, random_y, terrain_height + 0.5])
    
    # Führe Simulation für einige Schritte aus
    for step in range(100):
        environment.step(0.1)
        robot.update(0.1, environment)
        
        if step % 10 == 0:
            environment.render(mode='human')
    
    print(f"{TermColors.OKGREEN}Simulation abgeschlossen!{TermColors.ENDC}")


def main():
    """
    Hauptfunktion zum Starten der Simulation
    """
    # Kommandozeilenargumente parsieren
    import argparse
    parser = argparse.ArgumentParser(description="Komplexe Robotersimulation mit Multi-Sensor-Integration")
    
    parser.add_argument("--detail-level", type=int, choices=[1, 2, 3, 4], default=None,
                       help="Detailgrad der Simulation (1=niedrig, 4=ultra)")
    
    parser.add_argument("--train-steps", type=int, default=500,
                       help="Anzahl der Trainingsschritte")
    
    parser.add_argument("--test-steps", type=int, default=100,
                       help="Anzahl der Testschritte")
    
    parser.add_argument("--no-render", action="store_true",
                       help="Deaktiviere die Visualisierung (für Batch-Durchläufe)")
    
    parser.add_argument("--save-video", action="store_true",
                       help="Speichere die Simulation als Video")
    
    args = parser.parse_args()
    
    # Hardware-Erkennung
    hardware_manager = HardwareManager()
    simulation_config = hardware_manager.get_simulation_config()
    
    # Überschreibe Detailgrad, falls angegeben
    if args.detail_level is not None:
        simulation_config['physics_detail'] = args.detail_level
    
    # Umgebung erstellen
    env_config = {
        'size': (100, 100),
        'physics_detail_level': simulation_config['physics_detail'],
        'max_entities': simulation_config['max_entities']
    }
    
    environment = Environment(env_config)
    
    # Roboter erstellen
    robot_config = {
        'mass': 15.0,
        'dimension': np.array([0.5, 0.5, 0.3]),
        'sensor_detail_level': simulation_config['physics_detail']
    }
    
    robot = Robot(robot_config)
    
    # Setze Roboter auf eine gute Startposition
    terrain_height = environment.terrain.get_height(0, 0)
    robot.set_position([0, 0, terrain_height + 0.5])
    
    # Registriere den Roboter in der Umgebung
    environment.agents.append(robot)
    
    # Controller erstellen
    controller = create_robot_controller(robot, environment)
    
    # Trainingsdaten generieren
    print(f"Generiere Trainingsdaten mit {args.train_steps} Schritten...")
    X_train, y_train = generate_training_data(robot, environment, num_steps=args.train_steps)
    
    # Controller trainieren
    print("Trainiere neuronalen Controller...")
    controller.train((X_train, y_train), epochs=5, batch_size=32)
    
    # Controller speichern
    controller.save("robot_controller.h5")
    
    # Controller dem Roboter zuweisen
    robot.set_controller(controller)
    
    # Video-Aufnahme vorbereiten
    if args.save_video:
        import cv2
        frames = []
    
    # Testlauf durchführen
    print(f"Starte Testlauf mit {args.test_steps} Schritten...")
    
    # Setze den Roboter an eine neue Position
    random_x = np.random.uniform(-environment.config['size'][0]/3, environment.config['size'][0]/3)
    random_y = np.random.uniform(-environment.config['size'][1]/3, environment.config['size'][1]/3)
    terrain_height = environment.terrain.get_height(random_x, random_y)
    robot.set_position([random_x, random_y, terrain_height + 0.5])
    
    # Führe Simulation für festgelegte Anzahl Schritte aus
    for step in range(args.test_steps):
        environment.step(0.1)
        robot.update(0.1, environment)
        
        # Visualisierung
        if not args.no_render and (step % 10 == 0 or step == args.test_steps - 1):
            if args.save_video:
                frame = environment.render(mode='rgb_array')
                frames.append(frame)
            else:
                environment.render(mode='human')
    
    # Video speichern, falls gewünscht
    if args.save_video and frames:
        print("Speichere Simulationsvideo...")
        
        try:
            video_path = 'simulation_video.mp4'
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))
            
            for frame in frames:
                # OpenCV erwartet BGR-Format
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(bgr_frame)
            
            video.release()
            print(f"Video gespeichert unter: {video_path}")
        
        except Exception as e:
            print(f"Fehler beim Speichern des Videos: {e}")
    
    print("Simulation abgeschlossen!")


if __name__ == "__main__":
    # Starte die Simulation mit Standardparametern
    main()
