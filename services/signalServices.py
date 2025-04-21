import platform
from requests import get, RequestException, post
from requests.exceptions import RequestException
from json import dumps
from services.json_utils import read_json
from concurrent.futures import ThreadPoolExecutor
import sys
from cv2 import imencode
from os.path import abspath, join, dirname
sys.path.append(abspath(join(dirname(__file__), '..')))

def check_connection(url='https://www.google.com'):
    try:
        response = get(url)
        if response.status_code == 200:
            return True
        else:
            return False
    except RequestException as e:
        return False
    
def send_json_to_nodeMCU(known_people, nodeMCU_IP):
    url = nodeMCU_IP  + "/sync"
    json_data = dumps({"known_people": known_people}) 
    try:
        response = post(url, data=json_data, headers={'Content-Type': 'application/json'})
        print("Status Code:", response.status_code)
        print("Response Text:", response.text)
    except RequestException as e:
        print("Json post Connection error:", e)

def send_post_request(url, **kwargs):
    try:
        response = post(url, **kwargs)
        return response
    except RequestException as e:
        print(f"Hata: {e}")
        return None

def telegram_send_message(text, img):
    config_telegram = read_json("facefinder/config/telegram.json")
    try:
        payload = {
            'chat_id': config_telegram["CHAT_ID"]
        }
        with ThreadPoolExecutor(max_workers=2) as executor:
            if img is not None:
                _, img_encoded = imencode('.jpg', img)
                img_bytes = img_encoded.tobytes()
                files = {"photo": ("tlgrm.jpg", img_bytes, "image/jpeg")}
                
                # Caption ekleyerek fotoğraf gönder
                payload["caption"] = text
                future_image = executor.submit(send_post_request, f'https://api.telegram.org/bot{config_telegram["TOKEN"]}/sendPhoto', data=payload, files=files)
            else:
                payload["text"] = text
                future_text = executor.submit(send_post_request, f'https://api.telegram.org/bot{config_telegram["TOKEN"]}/sendMessage', data=payload, files=None)
    except Exception as e:
        print(f"message cant be sent: {e}")

def ping_device(ip,endpoint="checkDevice"):
    url = f"http://{ip}:600/{endpoint}"  # Replace 'your-endpoint' with the actual endpoint on your device
    try:
        response = get(url, timeout=10)  # You can adjust the timeout as needed
        
        if response.status_code == 200:
            return True
        else:
            return False
    except RequestException as e:
        print(f"Error contacting {ip}: {e}")
        return False

import netifaces

def get_local_ip_raspi():
    interfaces = netifaces.interfaces()
    for interface in interfaces:
        if interface != 'lo':  # Ignore loopback interface
            addresses = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addresses:
                return addresses[netifaces.AF_INET][0]['addr']
def get_local_ip_macos():
    for interface in netifaces.interfaces():
        if interface.startswith("en") or interface.startswith("eth"):  # Wi-Fi (en0), Ethernet (eth0)
            addresses = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addresses:
                return addresses[netifaces.AF_INET][0]['addr']

    return 

def get_raspi_ip():
    if platform.system() == "Linux":
        broker_IP = get_local_ip_raspi()
    elif platform.system() == "Darwin":
        broker_IP = get_local_ip_macos()
    return broker_IP