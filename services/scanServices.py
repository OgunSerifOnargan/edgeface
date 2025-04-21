from requests import get, RequestException
from ipaddress import ip_network
from concurrent.futures import ThreadPoolExecutor, as_completed
from json import loads, JSONDecodeError, dump

# Belirlenen IP aralığı ve port
IP_RANGE = '192.168.1.0/24'  # Bu IP aralığını kendi ağınıza göre güncelleyin
PORT = 600
ENDPOINT = '/checkUid?token=f436b49c-d482-4315-b7b9-c9c0297ec4c1'

# HTTP GET isteği atma ve yanıtı döndürme
def check_device(ip):
    url = f'http://{ip}:{PORT}{ENDPOINT}'
    try:
        response = get(url, timeout=6) 
        return ip, response.status_code, response.text
    except RequestException:
        return ip, None, None

# IP taraması yaparak cihazları kontrol etme
def scan_network(ip_range):
    devices = {}
    network = ip_network(ip_range)
    id_counter = 1
    with ThreadPoolExecutor(max_workers=128) as executor:
        futures = [executor.submit(check_device, str(ip)) for ip in network.hosts()]
        for future in as_completed(futures):
            ip, status_code, response_text = future.result()
            if status_code == 200 and response_text:
                print(ip, response_text)
                try:
                    response_json = loads(response_text)
                    if "station_id" in response_json:
                        key = "station_id"
                        value = response_json["station_id"]
                    elif "relay_id" in response_json:
                        key = "relay_id"
                        value = response_json["relay_id"]

                    devices[id_counter] = {
                        'ip': ip,
                        "id" : value,
                        'type': response_json.get('type', 'unknown')
                    }
                    id_counter += 1
                except JSONDecodeError:
                    print(f"IP {ip} adresinden dönen yanıt geçerli bir JSON değil: {response_text}")
    return devices

def group_by_box_id(devices):
    grouped_devices = {}
    for device_id, device_info in devices.items():
        box_id = device_info['box-id']
        if box_id not in grouped_devices:
            grouped_devices[box_id] = []
        grouped_devices[box_id].append(device_info)
    return grouped_devices

# Cihazları JSON dosyasına yazma
def write_to_json(devices, filename='devices.json'):
    with open(filename, 'w') as output_file:
        dump(devices, output_file, indent=4)