from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from signal import SIGKILL
import subprocess
from setproctitle import setproctitle
from gc import collect
from time import sleep, time
from json import dumps
from base64 import b64encode
from cv2 import CAP_FFMPEG, CAP_PROP_BUFFERSIZE, imencode, VideoCapture, imshow, waitKey, destroyAllWindows
from threading import Thread
from services.json_utils import read_json
from multiprocessing import Queue

class camera():
    def __init__(self, station_id, cam_IP):
        self.station_id = station_id
        self.cam_IP = cam_IP
        self.display_name = f"CAM {self.station_id}"
        self.frame_collection_queue = Queue(maxsize=1)
        self.faceDet_to_faceId_queue = Queue(maxsize=1)
        self.display_queue = Queue(maxsize=1)
        self.camIP_updater_queue = Queue(maxsize=1)
        self.frame_server_queue = Queue(maxsize=1)  # Ensure frames are fed into this queue
        self.streaming_port = 1500 + self.station_id  # Adjusted port 
        self.recorder_option = f"http://{self.cam_IP}:81/stream"
        self.current_frame = None  # Ensure frames are continuously updated
        self.hedef_saatler = {6, 14, 22}  # İşlem yapılacak saatler
        self.son_calisma_saati = time()  # En son çalıştığı saat
        self.config_system = read_json("facefinder/config/system.json")

    class FrameStreamHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == f'/cam{self.server.camera.station_id}':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()

                try:
                    frame = self.server.camera.frame_server_queue.get(block=True)
                    if frame is not None:
                        _, jpeg = imencode('.jpg', frame)
                        jpeg_bytes = jpeg.tobytes()
                        jpeg_base64 = b64encode(jpeg_bytes).decode('utf-8')
                        response = {"image": jpeg_base64}
                        json_response = dumps(response)
                        self.wfile.write(json_response.encode('utf-8'))
                except Exception as e:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(dumps({"error": str(e)}).encode('utf-8'))

    class CustomHTTPServer(HTTPServer):
        def __init__(self, server_address, RequestHandlerClass, camera_instance):
            super().__init__(server_address, RequestHandlerClass)
            self.camera = camera_instance

    def start_streaming(self):
        def run_server():
            server = self.CustomHTTPServer(('0.0.0.0', self.streaming_port), self.FrameStreamHandler, self)
            print(f"Streaming started at http://0.0.0.0:{self.streaming_port}/cam{self.station_id}")
            server.serve_forever()

        Thread(target=run_server, daemon=True).start()

    def connect_to_camera(self):
        print(f"Attempting to connect to CAM: {self.station_id}")
        if self.station_id == 2:
            cap = VideoCapture(self.recorder_option, CAP_FFMPEG)
            cap.set(CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = VideoCapture(self.recorder_option)
        if cap.isOpened():
            print(f"Connected to Cam {self.station_id}")
            return cap
        else:
            return None


    def collectFrames(self, broken_cam_id_list):
        def run_camera():
            if self.config_system["DEBUG_MODE"]:
                print(f"collect frames threadi basladı {self.station_id}")
            self.free_port()
            self.start_streaming()
            cap = self.connect_to_camera()
            frame_counter = 0
            while True:
                if cap is not None:
                    ret, frame = cap.read()
                    frame_counter+=1
                    if not ret:
                        self.config_system = read_json("facefinder/config/system.json")
                        collect()
                        print("Failed to read frame from camera")
                        cap.release()
                        cap = self.repair_cam(broken_cam_id_list)
                        continue
                    else:
                        if not self.frame_collection_queue.full():
                            self.frame_collection_queue.put(frame)
                        if not self.frame_server_queue.full():
                            self.frame_server_queue.put(frame)
                        if frame_counter%1000==0:
                            self.config_system = read_json("facefinder/config/system.json")
                            collect()
                else:
                    cap = self.repair_cam(broken_cam_id_list)
                # current_hour = datetime.now(pytz.timezone("Europe/Istanbul")).hour
                # if current_hour in self.hedef_saatler and self.son_calisma_saati != current_hour:
                #     restart_device(self.cam_IP)
                #     sleep(2)
                #     cap = self.repair_cam_with_same_IP()
                #     self.son_calisma_saati = current_hour


            # Start the frame collector thread
        collector_thread = Thread(target=run_camera, daemon=True)
        collector_thread.start()
            
    def displayFrames(self, stop_event, x):
        setproctitle(f"py_displayFrames_{self.station_id}")
        while not stop_event.is_set():
            display_frame = self.display_queue.get(block=True)
            imshow(self.display_name, display_frame)
            if waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
            
        destroyAllWindows()

    def repair_cam(self, broken_cam_id_list):
        while True:
            broken_cam_id_list.append(self.station_id)
            print(f"broken_cam_id_list e eklendi: {broken_cam_id_list}")
            self.cam_IP = self.camIP_updater_queue.get(block=True)
            self.recorder_option = f"http://{self.cam_IP}:81/stream"
            print(f"COLLECTFRAMES: New camera IP is received.")
            cap = self.connect_to_camera()
            if cap != None:
                print(f"Bir cam tamir edildi. : {self.station_id}")
                return cap
            else:
                print("Cam is not repaired. Retrying...")
                continue

    def repair_cam_with_same_IP(self):
        while True:
            self.recorder_option = f"http://{self.cam_IP}:81/stream"
            print(f"COLLECTFRAMES: Trying to reconnect with same IP...")
            cap = self.connect_to_camera()
            if cap != None:
                print(f"Bir cam restart sonrası tekrar bağlandı edildi. : {self.station_id}")
                return cap
            else:
                print("Cam is not repaired. Retrying...")
                continue

    def free_port(self):
        """ Belirtilen portu kullanan işlemi öldürerek portu serbest bırakır """
        try:
            while True:
                result = subprocess.run(["lsof", "-t", f"-i:{self.streaming_port}"], capture_output=True, text=True)
                pids = result.stdout.strip().split()

                if not pids:
                    if self.config_system["DEBUG_MODE"]:
                        print(f"Port {self.streaming_port} is now free.")
                    break  # Port boşsa çık

                for pid in pids:
                    if self.config_system["DEBUG_MODE"]:
                        print(f"Port {self.streaming_port} is in use by process {pid}. Terminating it...")
                    os.kill(int(pid), SIGKILL)

                sleep(5)  # Portun gerçekten serbest kalmasını bekle
        except Exception as e:
            print(f"Error freeing port {self.streaming_port}: {e}")

#     def connect_to_camera_UDP(self):
#         # if platform.system() == "Linux":
#         #     self.UDP_IP = get_local_ip_raspi()
#         # elif platform.system() == "Darwin":
#         #     self.UDP_IP = get_local_ip_macos()
#         self.UDP_IP = "0.0.0.0"
#         self.UDP_PORT = 2000 + self.station_id
#         self.CHUNK_LENGTH = 1460 
#         sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         sock.bind((self.UDP_IP, self.UDP_PORT))
#         print(f"UDP sunucusu {self.UDP_IP}:{self.UDP_PORT} adresinde dinleniyor...")
#         return sock
    
#     def collect_frames_UDP(self, stop_event, broken_cam_id_list):
#         free_port(self.streaming_port)
#         free_port(self.UDP_PORT)
#         self.start_streaming()
#         sock = self.connect_to_camera_UDP()
#         frame_counter = 0
#         malfunction_counter = 0
#         while True:
#             data = b""
#             while True:
#                 packet, addr = sock.recvfrom(1500)
#                 data += packet
#                 if len(packet) < self.CHUNK_LENGTH:
#                     break

#             nparr = frombuffer(data, uint8)
#             img = imdecode(nparr, IMREAD_COLOR)
#             if img is None:
#                 malfunction_counter+=1
#                 if malfunction_counter >= 5:
#                     self.repair_cam_UDP(broken_cam_id_list) #TODO: yaz
#             else:
#                 malfunction_counter = 0
#                 if not self.frame_collection_queue.full():
#                     self.frame_collection_queue.put(img)
#                 if not self.frame_server_queue.full():
#                     self.frame_server_queue.put(img)
#                 if frame_counter%1000==0:
#                     collect()
            
#     def repair_cam_UDP(self, broken_cam_id_list):
#         while True:
#             broken_cam_id_list.append(self.station_id)
#             print(f"broken_cam_id_list e eklendi: {broken_cam_id_list}")
#             self.cam_IP = self.camIP_updater_queue.get(block=True)
#             print(f"COLLECTFRAMES: New camera IP is received.")
#             sock = self.connect_to_camera_UDP()
#             if sock != None:
#                 print(f"Bir cam tamir edildi. : {self.station_id}")
#                 return sock
#             else:
#                 print("Cam is not repaired. Retrying...")
#                 continue

# # Belirlenen IP aralığı ve port
# PORT = 600
# ENDPOINT = '/restart?token=f436b49c-d482-4315-b7b9-c9c0297ec4c1'

# # HTTP GET isteği atma ve yanıtı döndürme
# def restart_device(ip):
#     url = f'http://{ip}:{PORT}{ENDPOINT}'
#     try:
#         response = get(url, timeout=6) 
#         return ip, response.status_code, response.text
#     except RequestException:
#         return ip, None, None