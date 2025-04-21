import os
import socket
import subprocess
from threading import Thread
from time import sleep
from signal import SIGKILL
import requests
from flask import Flask, jsonify, request
from platform import system
from classes.door import find_name_by_card_uid
from classes.face_recognizer import read_known_people_from_json_file
from services.json_utils import check_file_changed, read_json

class WebServer:
    def __init__(self, doors_ref, json_lock):
        """ Web server'ı başlatır ve gerekli verileri yükler """
        self.app = Flask(__name__)
        self.doors = doors_ref
        self.json_lock = json_lock
        self.config_system = read_json("facefinder/config/system.json")
        self.config_known_faces = read_json("facefinder/config/known_faces.json")
        self.local_ip = self.get_local_ip()
        # JSON verilerini yükle
        self.load_known_faces()
        # Flask endpointlerini tanımla
        self.setup_routes()

    def load_known_faces(self):
        """ Bilinen yüzleri JSON dosyasından yükler """
        station_id = "1" #TODO: otomatize et
        json_path = f'{self.config_known_faces["JSON_ROOT_PATH"]}/station_{station_id}/dlib_known_faces.json'
        _, self.known_face_names, _, self.known_card_uids = read_known_people_from_json_file(json_path, self.json_lock)

    def setup_routes(self):
        """ Flask endpointlerini tanımlar """
        @self.app.route('/accessInfo', methods=['POST'])
        def receive_data():
            data = request.get_json()
            if check_file_changed(f'{self.config_known_faces["JSON_ROOT_PATH"]}/station_{data["station_id"]}/dlib_known_faces.json'):
                self.load_known_faces()
            qr_json = {"123456789": "Ogun O.",
                       "234567891": "Uygar E.",
                       "345678912": "Burkay A."}
            printed_name = "Unknown"
            
            if data["type"] == "rfid":
                printed_name = find_name_by_card_uid(data["uuid"], self.known_card_uids, self.known_face_names)
            elif data["type"] == "qr":
                printed_name = qr_json.get(data["uuid"], "Unknown")
            elif data["type"] == "BLE" and data["uuid"] == "EventA":
                printed_name = "Ent_with_BLE"

            printed_name = printed_name.replace("_", " ")
            self.handle_door_access(data["station_id"], printed_name)

            return jsonify({'status': 'RFID card data is received.'}), 200

        @self.app.route('/findMyRP', methods=['GET'])
        def find_my_rp():
            try:
                hostname = socket.gethostname()
                ip_address = socket.gethostbyname(hostname)
                return jsonify({'ip_address': ip_address}), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/open_the_door', methods=['POST'])
        def open_door():
            try:
                data = request.get_json()
                self.open_the_door(data["relay_ip"])
                return jsonify({'status': 'Door has been opened.'}), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    def handle_door_access(self, station_id, printed_name):
        """ Kapının açılmasını yönetir """
        for adoor in self.doors.values():
            if printed_name != "Unknown":
                if adoor.station_indoor_id == int(station_id):
#                    adoor.station_indoor.init_mqtt_pub()
                    adoor.station_indoor.relay.post_result_to_relayDoor(adoor.station_indoor.mqtt_client)
                    adoor.station_indoor.lcd_control("Hos geldiniz", printed_name)
                    break
                elif adoor.station_outdoor_id == int(station_id):
                    adoor.station_outdoor.relay.post_result_to_relayDoor(adoor.station_outdoor.mqtt_client)
                    adoor.station_outdoor.lcd_control("Hos geldiniz", printed_name)
                    break
            else:
                if adoor.station_indoor_id == int(station_id):
                    adoor.station_indoor.lcd_control("Auth Rejected", "")
                elif adoor.station_outdoor_id == int(station_id):
                    adoor.station_outdoor.lcd_control("Auth Rejected", "")

    def get_local_ip(self):
        """ Cihazın yerel IP adresini alır """
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip_address = s.getsockname()[0]
            s.close()
            return ip_address
        except Exception:
            return "Unable to get IP"

    def free_port(self, port):
        """ Belirtilen portu kullanan işlemi öldürerek portu serbest bırakır """
        try:
            while True:
                result = subprocess.run(["lsof", "-t", f"-i:{port}"], capture_output=True, text=True)
                pids = result.stdout.strip().split()

                if not pids:
                    if self.config_system["DEBUG_MODE"]:
                        print(f"Port {port} is now free.")
                    break  # Port boşsa çık

                for pid in pids:
                    if self.config_system["DEBUG_MODE"]:
                        print(f"Port {port} is in use by process {pid}. Terminating it...")
                    os.kill(int(pid), SIGKILL)

                sleep(5)  # Portun gerçekten serbest kalmasını bekle
        except Exception as e:
            print(f"Error freeing port {port}: {e}")

    def run(self):
        """ Web server'ı çalıştırır """
        print(f"Server running on IP: {self.local_ip}")

        # HTTP ve HTTPS sunucularını ayrı thread'lerde başlat
        def run_http():
            self.free_port(2000)
            self.app.run(host=self.local_ip, port=2000)

        def run_https():
            self.free_port(3000)
            self.app.run(host=self.local_ip, port=3000, ssl_context=('server.crt', 'server.key'))

        http_thread = Thread(target=run_http)
        https_thread = Thread(target=run_https)

        http_thread.start()
        https_thread.start()

