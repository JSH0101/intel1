import os
import requests
import cv2
import numpy as np
import math
from PIL import Image
from requests.auth import HTTPBasicAuth
from io import BytesIO
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sqlite3
import uuid
from datetime import datetime

# Vision AI API 설정
VISION_API_URL = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/fc4c8e39-e27a-4f9f-b40f-b257ec663326/inference"
TEAM = "kdt2024_1-29"
ACCESS_KEY = "mHXlUqb4Nq3WHvF3eKuy22pKqqIIW4jLa4oCWLHx"

def create_table():
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS product (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        datetime TEXT,
        uuid TEXT UNIQUE,
        is_defective INTEGER,
        defect_reason TEXT
    )
    ''')
    conn.commit()
    conn.close()


def remove_left_corner_boxes(objects, x_threshold=5):
    """왼쪽 구석에 있는 박스 제거"""
    return [obj for obj in objects if obj['box'][2] > x_threshold]

def calculate_center(box):
    """박스 중심점 계산"""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy

def merge_boxes(box1, box2):
    """두 박스를 병합하여 하나의 박스로 만듦"""
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    return [x1, y1, x2, y2]

def remove_overlapping_boxes(objects, distance_threshold=20):
    """중심점 거리 기준으로 겹치는 박스를 병합 (스코어가 높은 박스 기준)"""
    objects = sorted(objects, key=lambda x: x.get('score', 0), reverse=True)
    filtered_objects = []
    for obj in objects:
        matched = False
        for filtered_obj in filtered_objects:
            if obj['class'] == filtered_obj['class'] and is_near(obj['box'], filtered_obj['box'], distance_threshold):
                filtered_obj['box'] = merge_boxes(obj['box'], filtered_obj['box'])
                matched = True
                break
        if not matched:
            filtered_objects.append(obj)
    return filtered_objects

def is_near(box1, box2, distance_threshold=20):
    """박스 중심점 간 거리 기준으로 겹침 여부 판단"""
    cx1, cy1 = calculate_center(box1)
    cx2, cy2 = calculate_center(box2)
    distance = math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
    return distance < distance_threshold

def filter_objects_by_count(objects):
    """HOLE은 4개, 나머지 클래스는 1개씩만 남김 (신뢰도 기준으로 필터링)"""
    objects = sorted(objects, key=lambda x: x.get('score', 0), reverse=True)
    filtered_objects = []
    counts = {}
    for obj in objects:
        label = obj["class"]
        score = obj.get("score", 0)
        # USB 클래스의 경우, 신뢰도가 0.79 이상일 때만 포함
        if label == "USB":
            if score >= 0.79:
                filtered_objects.append(obj)
                counts[label] = counts.get(label, 0) + 1
        # RASPBERRY PICO의 경우, 신뢰도가 0.8 이상일 때만 포함
        elif label == "RASPBERRY PICO" and score >= 0.8:
            filtered_objects.append(obj)
            counts[label] = counts.get(label, 0) + 1
        # BOOTSEL 클래스의 경우, 신뢰도가 0.85 이상일 때만 포함
        elif (label == "BOOTSEL"):
            if score >= 0.88:
                filtered_objects.append(obj)
                counts[label] = counts.get(label, 0) + 1
        # HOLE 클래스의 경우, 4개만 포함
        elif label != "RASPBERRY PICO" and score >= 0.4:
            if label == "HOLE" and counts.get(label, 0) < 4:
                filtered_objects.append(obj)
                counts[label] = counts.get(label, 0) + 1
            elif label != "HOLE" and counts.get(label, 0) < 1:
                filtered_objects.append(obj)
                counts[label] = counts.get(label, 0) + 1
    return filtered_objects, counts

current_window = None  # 현재 창 이름 저장

def process_image(image_path, nor_folder, un_folder):
    """이미지를 처리하고 라벨링된 결과 반환"""
    global current_window  # 현재 창 이름을 전역 변수로 사용

    # 이미지 읽기
    image = cv2.imread(image_path)

    # 원본 이미지 API 호출
    objects, counts = analyze_image(image)

    # 180도 회전한 이미지 API 호출
    rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    rotated_objects, _ = analyze_image(rotated_image)

    # 회전된 이미지에서 박스 좌표를 원본 이미지 기준으로 변환
    for obj in rotated_objects:
        x1, y1, x2, y2 = obj["box"]
        h, w, _ = image.shape
        obj["box"] = [w - x2, h - y2, w - x1, h - y1]

    # 원본과 회전된 결과를 통합
    objects.extend(rotated_objects)

    # 1단계: 왼쪽 구석에 있는 박스 제거
    objects = remove_left_corner_boxes(objects)

    # 2단계: 중심점 거리 기준으로 박스 겹침 제거
    objects = remove_overlapping_boxes(objects)

    # 3단계: HOLE=4, 나머지 클래스는 1개씩만 남기기 (신뢰도 필터 포함)
    objects, counts = filter_objects_by_count(objects)

    # 라벨링된 이미지 저장
    colors = {
        "RASPBERRY PICO": (255, 0, 0),
        "USB": (0, 255, 0),
        "CHIPSET": (0, 0, 255),
        "OSCILLATOR": (255, 255, 0),
        "BOOTSEL": (0, 255, 255),
        "HOLE": (255, 0, 255)
    }

    # 원본 이미지에 박스를 그리기
    for obj in objects:
        x1, y1, x2, y2 = obj["box"]
        label = obj["class"]
        confidence = obj.get("score", 0)
        box_color = colors.get(label, (0, 255, 0))
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
        label_text = f"{label} ({confidence:.2f})"
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

    # 최종적으로 결과를 저장
    output_path = os.path.join(nor_folder if is_nor(counts) else un_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, image)

    # 이전 창 닫기
    if current_window is not None:
        cv2.destroyWindow(current_window)

    # 새로운 창 열기
    current_window = "Processed Image"
    cv2.imshow(current_window, image)
    cv2.waitKey(0)  # 사용자가 키를 누를 때까지 대기

    # 데이터베이스에 결과 저장
    save_to_database(counts)

    return objects, counts

def analyze_image(image):
    """API를 호출하여 객체를 분석"""
    _, img_encoded = cv2.imencode('.jpg', image)
    response = requests.post(
        url=VISION_API_URL,
        auth=HTTPBasicAuth(TEAM, ACCESS_KEY),
        headers={"Content-Type": "image/jpeg"},
        data=img_encoded.tobytes(),
    )
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.text}")
        return [], {}
    
    results = response.json()
    return results.get("objects", []), {}

def is_nor(counts):
    """조건에 맞는지 검사 (HOLE=4, 나머지 각 1개, 총 9개)"""
    return counts.get("HOLE", 0) == 4 and \
           all(counts.get(key, 0) == 1 for key in ["RASPBERRY PICO", "USB", "CHIPSET", "OSCILLATOR", "BOOTSEL"]) and \
           sum(counts.values()) == 9

# 결과를 저장할 폴더 생성
def create_output_folders(base_path):
    nor_path = os.path.join(base_path, "nor")
    un_path = os.path.join(base_path, "un")
    os.makedirs(nor_path, exist_ok=True)
    os.makedirs(un_path, exist_ok=True)
    return nor_path, un_path

def save_to_database(counts):
    """
    데이터를 SQLite 데이터베이스에 저장. 
    조건에 따라 양품 또는 불량품 판정 및 결함 이유 추가.
    """
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()

    # 테이블 생성 (존재하지 않을 경우)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS product (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        datetime TEXT,
        uuid TEXT,
        is_defective INTEGER,
        defect_reason TEXT
    )
    ''')

    # 현재 시간 및 UUID 생성
    datetime_value = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    uuid_value = str(uuid.uuid4())

    # 결함 이유 작성
    defect_reason = []
    if counts.get("HOLE", 0) < 4:
        defect_reason.append(f"HOLE({counts.get('HOLE', 0)}/4)")
    for part in ["RASPBERRY PICO", "USB", "BOOTSEL"]:
        if counts.get(part, 0) < 1:
            defect_reason.append(f"{part}({counts.get(part, 0)}/1)")

    # 양품/불량품 판정
    if not defect_reason:  # 결함 이유가 없으면 양품
        is_defective = 0
        defect_reason_str = "normal"
    else:  # 결함 이유가 있으면 불량품
        is_defective = 1
        defect_reason_str = ", ".join(defect_reason)

    # 데이터베이스에 저장
    insert_query = '''
    INSERT INTO product (datetime, uuid, is_defective, defect_reason)
    VALUES (?, ?, ?, ?)
    '''
    cursor.execute(insert_query, (datetime_value, uuid_value, is_defective, defect_reason_str))
    conn.commit()
    conn.close()

# 폴더 내의 새로운 .jpg 파일이 추가되면 자동으로 처리하는 핸들러
class ImageHandler(FileSystemEventHandler):
    def __init__(self, nor_folder, un_folder):
        self.nor_folder = nor_folder
        self.un_folder = un_folder
    
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.jpg'):
            print(f"New image detected: {event.src_path}")
            process_image(event.src_path, self.nor_folder, self.un_folder)

# 폴더 경로 설정
folder_path = '/home/hyuna/test'
nor_folder, un_folder = create_output_folders(folder_path)

# watchdog을 사용해 폴더를 모니터링
event_handler = ImageHandler(nor_folder, un_folder)
observer = Observer()
observer.schedule(event_handler, folder_path, recursive=False)

# 모니터링 시작
observer.start()
print(f"Watching folder: {folder_path}")

try:
    while True:
        pass
except KeyboardInterrupt:
    observer.stop()
observer.join()
