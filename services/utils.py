from logging import INFO, getLogger, FileHandler, Formatter, DEBUG
import os
from requests import get, RequestException
from os.path import exists
from json import load, dump
from shelve import open

def xyxy_to_xywh(xyxy_box):
    """
    Convert a bounding box from (x_min, y_min, x_max, y_max) to (x, y, w, h).

    Parameters:
    xyxy_box (tuple): A tuple or list with 4 elements (x_min, y_min, x_max, y_max).

    Returns:
    tuple: A tuple with 4 elements (x, y, w, h).
    """
    x_min, y_min, x_max, y_max = xyxy_box
    x = x_min
    y = y_min
    w = x_max - x_min
    h = y_max - y_min
    return [x, y, w, h]

def get_biggest_bbox(bboxes):
    bboxes = bboxes.tolist()
    if len(bboxes) == 0:
        return []

    def bbox_area(bbox):
        x_min, y_min, x_max, y_max = bbox
        return (x_max - x_min) * (y_max - y_min)

    biggest = max(bboxes, key=bbox_area)
    return biggest

def xywh_to_xyxy(xywh_box):
    x, y, w, h = xywh_box
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h
    return (x_min, y_min, x_max, y_max)

def convert_xywh_to_xyxy(bbox_face_proposal):
    x1 = bbox_face_proposal['facial_area']['x']
    y1 = bbox_face_proposal['facial_area']['y']
    x2 = x1 + bbox_face_proposal['facial_area']['w']
    y2 = y1 + bbox_face_proposal['facial_area']['h']
    return [x1, y1, x2, y2]

def yolo_to_top_right_bottom_left(bbox_face_proposal):
    x_min, y_min, x_max, y_max = bbox_face_proposal
    top = int(y_min)
    right = int(x_max)
    bottom = int(y_max)
    left = int(x_min)
    return (top, right, bottom, left)

from logging import getLogger, FileHandler, Formatter, INFO

def initialize_logger(name, level=INFO):
    logger = getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        file_handler = FileHandler(name + ".log")
        formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False 
    return logger


def set_logger_level(name, level):
    logger = getLogger(name)
    logger.setLevel(level)

def disable_logger(name):
    logger = getLogger(name)
    logger.disabled = True

def enable_logger(name, level=DEBUG):
    logger = getLogger(name)
    logger.disabled = False
    logger.setLevel(level)

def crop_and_set_img_faceProposal_yolo(frame, bbox_face_proposal):
    x_min, y_min, x_max, y_max = [int(coord) for coord in bbox_face_proposal]
    img_face = frame[y_min:y_max, x_min:x_max]
    return img_face

def get_uid_name(ip):
    url = f"http://{ip}/check-uid"
    try:
        response = get(url, timeout=1/100)
        if response.status_code == 200:
            data = response.json()
            uid = data.get("uid")
            name = data.get("name")
            return uid, name
    except RequestException:
        return None, None

def scan_networks():
    ip_base = ["192.168.1."]
    devices = []

    for base in ip_base:
        for i in range(256):
            if (i >= 0 and i <= 255):
                ip = f"{base}{i}:600"
                uid, name = get_uid_name(ip)
                print(ip, "\n")
                if uid and name:
                    print(uid, name, ip)
                    devices.append({"ip": ip, "uid": uid, "name": name})
                    if len(devices) >= 2:
                        break

    return devices

def save_to_json_file(devices, filename="devices.json"):
    if exists(filename):
        with open(filename, "r") as file:
            existing_devices = load(file)
    else:
        existing_devices = []

    existing_devices.extend(devices)

    with open(filename, "w") as file:
        dump(existing_devices, file, indent=4)

def scanning_and_saving():
    devices = scan_networks()

    if devices:
        save_to_json_file(devices)
        print(f"{len(devices)} yeni cihaz bilgisi 'devices.json' dosyasına eklendi.")
    else:
        print("Hiçbir cihaz bulunamadı.")

def match_and_write():
    scanning_and_saving()
    # JSON dosyasını oku
    json_file_path = "devices.json"

    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = load(json_file)

    paired_devices = {}

    # Cihazları eşleştir
    for device in data:
        uid = device["uid"]
        name = device["name"]

        if uid not in paired_devices:
            paired_devices[uid] = []
        paired_devices[uid].append(device)

    # global_functions\app_constants.py dosyasını oku
    constants_file_path = "global_functions/app_constants.py"

    with open(constants_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Cihazları eşleşen uid'lere göre güncelle
    index = 1
    for uid, devices in paired_devices.items():
        if len(devices) > 1:
            cam_device = next((d for d in devices if d["name"] == "Cam"), None)
            nodemcu_device = next((d for d in devices if d["name"] == "Nodemcu"), None)
            if cam_device and nodemcu_device:
                print(f"UID {uid} ile eşleşen cihazlar:")
                print(f"Cam: {cam_device}")
                print(f"Nodemcu: {nodemcu_device}")

                # IP adreslerini güncelle
                for i, line in enumerate(lines):
                    if index == 1 and "nodeMCU_IP1" in line:
                        lines[i] = f'nodeMCU_IP1 = "http://{nodemcu_device["ip"]}:601"\n'
                    if index == 1 and "esp32Camera_IP_1" in line:
                        lines[i] = f'esp32Camera_IP_1 = "http://{cam_device["ip"]}:81/stream"\n'
                    if index == 2 and "nodeMCU_IP2" in line:
                        lines[i] = f'nodeMCU_IP2 = "http://{nodemcu_device["ip"]}:601"\n'
                    if index == 2 and "esp32Camera_IP_2" in line:
                        lines[i] = f'esp32Camera_IP_2 = "http://{cam_device["ip"]}:81/stream"\n'
                index += 1

    # Dosyayı güncelle
    with open(constants_file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)

    print("IP adresleri başarıyla güncellendi.")

def on_press(key, stop_event):
    try:
        if key.char == 's':
            print("Stopping all processes...")
            stop_event.set()
    except AttributeError:
        pass

# hardware queue utils
def save_to_shelve(queue_name, info):
    with open(queue_name) as db:
        if 'queue' not in db:
            db['queue'] = []
        db['queue'].append(info)

def load_from_shelve(queue_name):
    with open(queue_name) as db:
        if 'queue' in db:
            queue = db['queue']
            db['queue'] = queue[1:]
            return queue[0]
        else:
            return None

def fullname_to_printedname(full_name):
    if len(full_name.split("_")) == 2:
        name = full_name.split("_")[0]
        surname = full_name.split("_")[1]
        printed_name = name + " " + surname[0] + "."
        return printed_name
    else:
        return full_name
