import json
from re import match
import shelve
from threading import Thread
from classes.landmark_detector import landmark_detector
from services.mqtt_listener import MQTTClient
from processes.image_processing import preprocess_face_img
from classes.nodemcu import nodemcu
from classes.camera import camera
from classes.result_person_info import result_person_info
from classes.face import face
from classes.relay import relay
from services.signalServices import check_connection, get_local_ip_macos, get_local_ip_raspi, get_raspi_ip, telegram_send_message
from services.utils import fullname_to_printedname
from services.json_utils import read_json, check_file_changed
import platform
from datetime import datetime
from setproctitle import setproctitle
from time import sleep, time
from multiprocessing import Lock, Queue
from supervision import Detections, ByteTrack
from os import environ, makedirs, listdir, remove, listdir
from os.path import exists, join, getmtime, splitext
from warnings import filterwarnings
from gc import collect
from numpy import array
from cv2 import QRCodeDetector, imshow, imwrite, waitKey
from psutil import Process as psutil_process
from os import getpid
from copy import deepcopy
import paho.mqtt.client as mqtt
from threading import Thread, Lock
import shelve
environ['GLOG_minloglevel'] = '2'
filterwarnings("ignore", category=UserWarning)
# from sklearn.decomposition import PCA
import logging
logging.basicConfig(
    level=logging.INFO,  # Log seviyesini belirle
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log formatı
    datefmt="%Y-%m-%d %H:%M:%S",  # Tarih formatı
    filemode="a"
)

class station():
    def __init__(self, station_id, cam_IP, nodeMCU_IP, relay_id, IP, landmark_queue_dict, face_recognizer_queue_dict, faceDet_to_landmark_queue, faceID_to_faceRecognizer_queue, json_lock):
            self.landmark_queue_dict = landmark_queue_dict
            self.station_id = int(station_id)
            self.faceDet_to_landmark_queue = faceDet_to_landmark_queue
            self.landmark_to_faceDet_queue = landmark_queue_dict[f"{self.station_id}"]["landmark_to_faceDet_queue"]
            self.faceID_to_faceRecognizer_queue = faceID_to_faceRecognizer_queue
            self.faceRecognizer_to_faceID_queue = face_recognizer_queue_dict[f"{self.station_id}"]["faceRecognizer_to_faceID_queue"]
            self.cam_IP = cam_IP
            self.nodeMCU_IP = nodeMCU_IP
            self.nodeMCU = nodemcu(self.station_id, nodeMCU_IP)
            self.camera = camera(self.station_id, cam_IP)
            self.relay = relay(relay_id, IP)
            self.relay_IP_updater_queue = Queue(maxsize=1)
            self.broker_IP = get_raspi_ip()
            self.byte_tracker = ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30, minimum_matching_threshold=0.8, frame_rate=5)
            self.faces = {}
            self.found_face_dict = {}
            self.last_unknown_time = 0
            self.shelve_lock = Lock()
            self.json_lock = json_lock
            self.last_analiz_ediliyor_time = 0
            self.qr_reader = QRCodeDetector()
            self.logger = logging.getLogger(__name__)
            self.config_known_faces = read_json("facefinder/config/known_faces.json")
            self.config_system = read_json("facefinder/config/system.json")
            self.shelve_lock = Lock()
            # landmarkDetector = landmark_detector(usage_reason="realtime",
            #                              landmark_queue_dict=landmark_queue_dict,
            #                             up_threshold=0.5,
            #                             down_threshold=2,
            #                             left_threshold=3,
            #                             right_threshold=0.45,
            #                             display_mode=DISPLAY_MODE)        
            # faceDet_to_landmark_queue = landmarkDetector.faceDet_to_landmark_queue
            # landmarkDetector.faceLandmarkDet()

    def station_main_loop(self, stop_event, broken_cam_id_list):
        setproctitle(f"py_station_main_loop{self.station_id}") 
        process = psutil_process(getpid())
        system = platform.system()
        if system == "Linux":
            if self.station_id == 1:
                process.cpu_affinity([0])
            else:
                process.cpu_affinity([1])
        self.mqtt_client = MQTTClient("station")
        self.mqtt_client.start()

        self.camera.collectFrames(broken_cam_id_list)
        self.faceDet()
        self.faceID()
        while not stop_event.is_set():
            #config changes check
            if check_file_changed("facefinder/config/known_faces.json"):
                self.config_known_faces = read_json("facefinder/config/known_faces.json")
            if check_file_changed("facefinder/config/system.json"):
                self.config_system = read_json("facefinder/config/system.json")
            #IP changes checks
            if not self.relay_IP_updater_queue.empty():
                self.relay.IP = self.relay_IP_updater_queue.get()
                print(f"relay_IP is updated in station: {self.station_id}")
            if not self.nodeMCU.nodeMCU_IP_updater_queue.empty():
                self.nodeMCU_IP = self.nodeMCU.nodeMCU_IP_updater_queue.get()
                self.nodeMCU.nodeMCU_IP = self.nodeMCU_IP 
                print(f"nodemcu_IP is updated in station: {self.station_id}")

            sleep(1)

    def faceDet(self):
        def face_detection_loop():
            if self.config_system["DEBUG_MODE"]:
                print(f"facedet kodu çalışmaya başladı {self.station_id}")
            face_not_found_counter = 0
            frame_counter = 0
            landmarkDetector = landmark_detector(usage_reason="realtime",
                                         landmark_queue_dict=self.landmark_queue_dict,
                                         station_id=self.station_id,
                                        up_threshold=0.5,
                                        down_threshold=2,
                                        left_threshold=3,
                                        right_threshold=0.45,
                                        display_mode=self.config_system["DISPLAY_MODE"])  
            while True:
                frame = self.camera.frame_collection_queue.get(block=True)
                # print(f"faceDet is processing frames for {self.station_id}")
                defaultFrame = frame.copy()
                if self.config_system["DISPLAY_MODE"]:
                    imshow(f'{self.station_id}', defaultFrame)
                    waitKey(1)
                frame_counter+=1
                image, face_img, bbox_biggest_xyxy, flag_face_skip, user_warning1, user_warning2, mov_mag = landmarkDetector.crop_face_from_landmarks(frame)
                # self.faceDet_to_landmark_queue.put([frame, self.station_id])
                # image, face_img, bbox_biggest_xyxy, flag_face_skip, user_warning1, user_warning2, mov_mag = self.landmark_to_faceDet_queue.get(block=True)
                if face_img is not None:
                    if time() - self.last_analiz_ediliyor_time > 5:
                        self.lcd_control("Analiz ediliyor", "Lutfen bekleyin...")
                        self.last_analiz_ediliyor_time = time()
                    # print(f"faceDET {self.station_id}: face is found")
                    face_img = preprocess_face_img(face_img)
                    face_not_found_counter = 0
                    self.camera.faceDet_to_faceId_queue.put([defaultFrame, bbox_biggest_xyxy, face_img])
                else:
                    face_not_found_counter += 1

                if face_not_found_counter == 1000:
                    #print(f"faceDET {self.station_id}: dummy img is send to queue")
                    self.camera.faceDet_to_faceId_queue.put([None, None, None])
                    self.failed_post_to_OMEGA()
                    collect()
                    #print(f"faceDET {self.station_id}: gc collected")
                    face_not_found_counter = 0
                if frame_counter % 3 == 0:
                    frame_counter = 0
                    try:
                        data, qr_bbox, _ = self.qr_reader.detectAndDecode(defaultFrame)
                        if qr_bbox is not None:
                            if data:
                                if self.config_system["DEBUG_MODE"]:
                                    print(data.split("\n")[0])
                                self.relay.post_result_to_relayDoor(self.mqtt_client)
                                if self.config_system["DEBUG_MODE"]:
                                    telegram_send_message(data, None)
                                self.lcd_control("Hos geldiniz", f"QR: {data}")
                    except Exception as e:
                        print(e, "QR okunamadı")
        faceDet_thread = Thread(target=face_detection_loop, daemon=True)
        faceDet_thread.start()

    def faceID(self):
        def faceID_loop():
            if self.config_system["DEBUG_MODE"]:
                print(f"faceid kodu calısmaya basladı {self.station_id}")
            while True:
                defaultFrame, bbox_biggest_xyxy, face_img = self.camera.faceDet_to_faceId_queue.get(block=True)
                if defaultFrame is not None:
                    st = time()
                    self.defaultFrame = defaultFrame
                    detections = self.update_byteTracker(bbox_biggest_xyxy)
                    self.faceID_to_faceRecognizer_queue.put([face_img, "dlib", self.station_id])
                    dlib_encodedVector, dlib_result = self.faceRecognizer_to_faceID_queue.get(block=True)

                    self.logger.info("dlib results: %s, %s, %s, %s ", dlib_result["dlib_first_min_key"], dlib_result["dlib_first_min_val"], dlib_result["dlib_second_min_key"], dlib_result["dlib_second_min_val"])
                    if detections.tracker_id.size > 0:
                        if dlib_result["dlib_first_min_key"] != "":
                            isnt_same_face_within_5_sec, full_name, printed_name = self.is_same_face_within_5_sec(dlib_result)  
                            if isnt_same_face_within_5_sec:
                                if detections.tracker_id[-1] not in self.faces.keys():
                                    print(f"new face is created: {detections.tracker_id}")
                                    self.faces[detections.tracker_id[-1]] = face(detections.tracker_id[-1], face_img, st, dlib_encodedVector, full_name, printed_name, dlib_result)
                                else:
                                    self.faces[detections.tracker_id[-1]].update_dlib_results(dlib_encodedVector, face_img, full_name, printed_name, dlib_result)
                                self.faces[detections.tracker_id[-1]].update_bestImg_dist_pair(self.defaultFrame)
                                img_count_of_the_person = count_images_in_folder(f'{self.config_known_faces["KNOWN_FACES_ROOT_DIR"]}/station_{self.station_id}/{full_name}')
                                if self.faces[detections.tracker_id[-1]].temp_dlib_result["dlib_first_min_val"] <= 0.33:
                                    self.faces[detections.tracker_id[-1]].finalizer_model = "dlib"
                                    self.faces[detections.tracker_id[-1]].finalize_face_info("dlib")
                                    current_face = deepcopy(self.faces[detections.tracker_id[-1]])
                                    self.faces[detections.tracker_id[-1]].ready_to_finalize = True

                                elif self.faces[detections.tracker_id[-1]].temp_dlib_result["dlib_first_min_val"] > 0.52:
                                    self.faces[detections.tracker_id[-1]].update_counters(5, 0)

                                elif self.faces[detections.tracker_id[-1]].temp_dlib_result["dlib_first_min_val"] >= 0.33 and self.faces[detections.tracker_id[-1]].temp_dlib_result["dlib_first_min_val"] < 0.52:
                                    if img_count_of_the_person <= 1:
                                        self.logger.info("İlk defa kamera karşısında...")
                                        #self.lcd_control("Ilk kayit aliniyor", "Lutfen bekleyin...")
                                        
                                        if self.faces[detections.tracker_id[-1]].temp_dlib_result["dlib_first_min_val"]>=0.33 and self.faces[detections.tracker_id[-1]].temp_dlib_result["dlib_first_min_val"]<0.42:
                                            self.faces[detections.tracker_id[-1]].update_counters(0, 1)
                                        elif self.faces[detections.tracker_id[-1]].temp_dlib_result["dlib_first_min_val"]>=0.42 and self.faces[detections.tracker_id[-1]].temp_dlib_result["dlib_first_min_val"]<0.49:
                                            self.faces[detections.tracker_id[-1]].update_counters(None, 2)
                                        elif self.faces[detections.tracker_id[-1]].temp_dlib_result["dlib_first_min_val"]>=0.49 and self.faces[detections.tracker_id[-1]].temp_dlib_result["dlib_first_min_val"]<0.52:  
                                            self.faces[detections.tracker_id[-1]].update_counters(3, 3) 
                                    elif img_count_of_the_person > 1:
                                        if self.faces[detections.tracker_id[-1]].temp_dlib_result["dlib_first_min_val"]>0.33 and self.faces[detections.tracker_id[-1]].temp_dlib_result["dlib_first_min_val"]<0.40:
                                            self.faces[detections.tracker_id[-1]].update_counters(0, 1)
                                        elif self.faces[detections.tracker_id[-1]].temp_dlib_result["dlib_first_min_val"]>=0.40 and self.faces[detections.tracker_id[-1]].temp_dlib_result["dlib_first_min_val"]<0.44:
                                            self.faces[detections.tracker_id[-1]].update_counters(0, 2)
                                        elif self.faces[detections.tracker_id[-1]].temp_dlib_result["dlib_first_min_val"]>=0.44 and self.faces[detections.tracker_id[-1]].temp_dlib_result["dlib_first_min_val"]<0.46:
                                            self.faces[detections.tracker_id[-1]].update_counters(None, 3)
                                        elif self.faces[detections.tracker_id[-1]].temp_dlib_result["dlib_first_min_val"]>=0.46 and self.faces[detections.tracker_id[-1]].temp_dlib_result["dlib_first_min_val"]<0.48:
                                            self.faces[detections.tracker_id[-1]].update_counters(None, 4)
                                        elif self.faces[detections.tracker_id[-1]].temp_dlib_result["dlib_first_min_val"]>=0.48 and self.faces[detections.tracker_id[-1]].temp_dlib_result["dlib_first_min_val"]<0.52:
                                            pass

                                if self.faces[detections.tracker_id[-1]].unknown_counter >= 30:#TODO: Optimize edilecek
                                    print("unknown is attained to trackerID: {}".format(detections.tracker_id[-1]))
                                    self.faces[detections.tracker_id[-1]].temp_printed_name = "Unknown"
                                    self.faces[detections.tracker_id[-1]].final_printed_name = "Unknown"
                                    self.faces[detections.tracker_id[-1]].finalizer_model = "dlib"
                                    self.faces[detections.tracker_id[-1]].finalize_face_info("dlib")
                                    current_face = deepcopy(self.faces[detections.tracker_id[-1]])
                                    self.faces[detections.tracker_id[-1]].ready_to_finalize = True 

                                elif self.faces[detections.tracker_id[-1]].huge_model_trigger_counter >= 12: 
#                                    self.lcd_control("Facenet calisiyor", "Lutfen bekleyin...")
                                    self.faceID_to_faceRecognizer_queue.put([self.faces[detections.tracker_id[-1]].bestImg_dist_pair["img"], "facenet", self.station_id])
                                    self.faces[detections.tracker_id[-1]].temp_facenet_encodedVector, self.faces[detections.tracker_id[-1]].temp_facenet_result = self.faceRecognizer_to_faceID_queue.get(block=True)
                                    self.logger.info("facenet results:  %s, %s, %s, %s ", self.faces[detections.tracker_id[-1]].temp_facenet_result["facenet_first_min_key"], self.faces[detections.tracker_id[-1]].temp_facenet_result["facenet_first_min_val"], self.faces[detections.tracker_id[-1]].temp_facenet_result["facenet_second_min_key"], self.faces[detections.tracker_id[-1]].temp_facenet_result["facenet_second_min_val"])
                                    print(f'facenet minkey/minval : {self.faces[detections.tracker_id[-1]].temp_facenet_result["facenet_first_min_key"]} : {self.faces[detections.tracker_id[-1]].temp_facenet_result["facenet_first_min_val"]}')
                                    if self.faces[detections.tracker_id[-1]].temp_facenet_result["facenet_first_min_val"] <= 21 and self.faces[detections.tracker_id[-1]].bestImg_dist_pair["min_val"] <= 0.45 and self.faces[detections.tracker_id[-1]].temp_facenet_result["facenet_first_min_key"] == self.faces[detections.tracker_id[-1]].bestImg_dist_pair["min_key"]:
                                        self.faces[detections.tracker_id[-1]].finalizer_model = "facenet"
                                        self.faces[detections.tracker_id[-1]].finalize_face_info("facenet")
                                        current_face = deepcopy(self.faces[detections.tracker_id[-1]])
                                        self.faces[detections.tracker_id[-1]].ready_to_finalize = True 
                                    elif self.faces[detections.tracker_id[-1]].temp_facenet_result["facenet_first_min_val"] <= 19 and self.faces[detections.tracker_id[-1]].bestImg_dist_pair["min_val"] >= 0.45 and self.faces[detections.tracker_id[-1]].bestImg_dist_pair["min_val"] <= 0.49 and self.faces[detections.tracker_id[-1]].temp_facenet_result["facenet_first_min_key"] == self.faces[detections.tracker_id[-1]].bestImg_dist_pair["min_key"]:
                                        self.faces[detections.tracker_id[-1]].finalizer_model = "facenet"
                                        self.faces[detections.tracker_id[-1]].finalize_face_info("facenet")
                                        current_face = deepcopy(self.faces[detections.tracker_id[-1]])
                                        self.faces[detections.tracker_id[-1]].ready_to_finalize = True 
                                    else:
                                        print("unknown is attained to trackerID: {}".format(detections.tracker_id[-1]))
                                        self.faces[detections.tracker_id[-1]].temp_printed_name = "Unknown"
                                        self.faces[detections.tracker_id[-1]].final_printed_name = "Unknown"
                                        self.faces[detections.tracker_id[-1]].finalizer_model = "facenet"
                                        self.faces[detections.tracker_id[-1]].finalize_face_info("facenet")
                                        current_face = deepcopy(self.faces[detections.tracker_id[-1]])
                                        self.faces[detections.tracker_id[-1]].ready_to_finalize = True 
                                
                                if self.faces[detections.tracker_id[-1]].ready_to_finalize:
                                    self.finalize_face(current_face)

        faceID_thread = Thread(target=faceID_loop, daemon=True)
        faceID_thread.start()

    def lcd_control(self, message1, message2):
        def lcd_control_in_thread():
            if message1 != "":
                topic = f"nodemcu_{self.station_id}/printLcd"  # Konu başlığı
                data = {"1:": message1, "2:": message2}
                
                try:
                    # MQTT ile mesaj gönder
                    payload = json.dumps(data)  # JSON formatında veri
                    self.mqtt_client.publish(topic, payload)  # Mesajı yayımla
                    
                except Exception as e:
                    print(f"Error sending MQTT message: {e}")

        thread = Thread(target=lcd_control_in_thread)
        thread.start()

    def finalize_face(self, current_face):
        if current_face.finalizer_model == "dlib":
            min_val = current_face.final_dlib_result["dlib_first_min_val"]
        elif current_face.finalizer_model == "facenet":
            min_val = current_face.final_facenet_result["facenet_first_min_val"]
        self.lcd_control("Hos geldiniz", f"{current_face.final_printed_name}")
        self.last_analiz_ediliyor_time = time()
        if not current_face.final_printed_name.lower().startswith("unknown"):
            self.relay.post_result_to_relayDoor(self.mqtt_client)
        print(f'{current_face.final_printed_name}    NEW FACE IS FOUND!!!')
        current_face.calculate_finalization_time()
        current_face_final = deepcopy(current_face)
        self.send_process(current_face_final)
        self.found_face_dict = {current_face_final.final_printed_name : time()}
        for i in range(3):
            garbage = self.camera.faceDet_to_faceId_queue.get(block=True)
        self.faces = {}
        self.byte_tracker = ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30, minimum_matching_threshold=0.8, frame_rate=6)
        collect()

    def send_process(self, current_face):
        def send_process_in_thread():
            helmet_state = False
            # helmet_state = self.ppe_detector.predict_PPE(self.defaultFrame)
            if current_face.final_printed_name == "Unknown":
                if current_face.finalizer_model == "dlib":
                    self.faceID_to_faceRecognizer_queue.put([current_face.bestImg_dist_pair["img"], "facenet", self.station_id])
                    current_face.final_facenet_encodedVector, facenet_result = self.faceRecognizer_to_faceID_queue.get(block=True)
                current_face.final_full_name = add_entry_to_json(current_face.final_dlib_encodedVector, "dlib", self.station_id, self.json_lock, self.config_known_faces["JSON_ROOT_PATH"])
                current_face.final_full_name = add_entry_to_json(current_face.final_facenet_encodedVector, "facenet", self.station_id, self.json_lock, self.config_known_faces["JSON_ROOT_PATH"])
                self.save_new_unknown_image(current_face)
                self.faceID_to_faceRecognizer_queue.put([None, None, self.station_id])
                if self.config_system["DEBUG_MODE"]:
                    telegram_send_message(f'Taninmayan kisi: {current_face.final_full_name} \n exe_time: {current_face.finalization_time:.2f} \n dlib_best_min_key:val : {current_face.bestImg_dist_pair["min_val"]} : {current_face.bestImg_dist_pair["min_key"]} \n\n dlib_first_min_key:val : {current_face.final_dlib_result["dlib_first_min_key"]} : {current_face.final_dlib_result["dlib_first_min_val"]} \n dlib_second_min_key:val : {current_face.final_dlib_result["dlib_second_min_key"]} : {current_face.final_dlib_result["dlib_second_min_val"]} \n\n facenet_first_min_key:val : {current_face.final_facenet_result["facenet_first_min_key"]} : {current_face.final_facenet_result["facenet_first_min_val"]},\n facenet_second_min_key:val : {current_face.final_facenet_result["facenet_second_min_key"]} : {current_face.final_facenet_result["facenet_second_min_val"]} \n {helmet_state}',
                                current_face.bestImg_dist_pair["defaultFrame"]
                                )
            else:             
                img_got_today, image_files, user_folder = self.check_user_folder(current_face.final_full_name)
                if not img_got_today:
                    if current_face.finalizer_model == "dlib":
                        self.faceID_to_faceRecognizer_queue.put([current_face.bestImg_dist_pair["img"], "facenet", self.station_id])
                        current_face.final_facenet_encodedVector, facenet_result = self.faceRecognizer_to_faceID_queue.get(block=True)
                    self.save_image_to_user_folder(image_files, user_folder, current_face)
                    self.update_encoding_in_json(current_face.final_full_name, current_face.final_dlib_encodedVector, "dlib", self.json_lock)
                    self.update_encoding_in_json(current_face.final_full_name, current_face.final_facenet_encodedVector, "facenet", self.json_lock)
                    self.faceID_to_faceRecognizer_queue.put([None, None, self.station_id])
                if self.config_system["DEBUG_MODE"]:
                    telegram_send_message(f'Giriş yapan : {current_face.final_full_name} \n exe_time: {current_face.finalization_time:.2f} \n \n dlib_best_min_key:val : {current_face.bestImg_dist_pair["min_val"]} : {current_face.bestImg_dist_pair["min_key"]} \n\n dlib_first_min_key:val : {current_face.final_dlib_result["dlib_first_min_key"]} : {current_face.final_dlib_result["dlib_first_min_val"]} \n dlib_second_min_key:val : {current_face.final_dlib_result["dlib_second_min_key"]} : {current_face.final_dlib_result["dlib_second_min_val"]} \n\n facenet_first_min_key:val : {current_face.final_facenet_result["facenet_first_min_key"]} : {current_face.final_facenet_result["facenet_first_min_val"]},\n facenet_second_min_key:val : {current_face.final_facenet_result["facenet_second_min_key"]} : {current_face.final_facenet_result["facenet_second_min_val"]} \n {helmet_state}', 
                                current_face.bestImg_dist_pair["defaultFrame"]
                                )

            info = result_person_info(self.station_id, current_face)
            self.post_face_result_to_OMEGA(info)

        thread = Thread(target=send_process_in_thread)
        thread.start()

    def check_user_folder(self, full_name):
        user_folder = f'{self.config_known_faces["KNOWN_FACES_ROOT_DIR"]}/station_{self.station_id}/{full_name}'
        if not exists(user_folder):
            makedirs(user_folder)

        image_files = sorted(
            [join(user_folder, f) for f in listdir(user_folder) if '_' not in f and " " not in f],
            key=getmtime
        )
        if image_files:
            latest_image_path = image_files[-1]
            latest_image_date = datetime.fromtimestamp(getmtime(latest_image_path)).date()
            current_date = datetime.now().date()

            if latest_image_date == current_date:
                print(f"An image for {full_name} has already been taken today.")
                return True, image_files, user_folder
            else:
                return False, image_files, user_folder
        else:
            return False, image_files, user_folder

    def save_image_to_user_folder(self,image_files, user_folder, current_face):
        if len(image_files) >= self.config_known_faces["MAX_IMAGES_PER_USER"]:
            remove(image_files[0])  # Remove the earliest image
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        image_path = join(user_folder, f"{timestamp}.jpg")
        imwrite(image_path, current_face.bestImg_dist_pair["img"])
        print(f"Saved new image for {current_face.final_full_name} at {image_path}")

    def save_new_unknown_image(self, current_face):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        user_folder = f'{self.config_known_faces["KNOWN_FACES_ROOT_DIR"]}/station_{self.station_id}/{current_face.final_full_name}'
        image_path = join(user_folder, f"{timestamp}.jpg")
        imwrite(image_path, current_face.bestImg_dist_pair["img"])
        print(f"Saved new image for {current_face.final_full_name} at {image_path}")
        
    def get_fullName_and_printedName(self, min_key):
        full_name = min_key
        printed_name = fullname_to_printedname(full_name)
        return full_name, printed_name

    def post_face_result_to_OMEGA(self, info):
        def post_face_result_to_OMEGA_in_thread():
            info.construct_body_for_post()
            try:
                info.send_post_to_db(info)
            except Exception as e:
                self.save_to_shelve('failed_post_queue', info)
        thread = Thread(target=post_face_result_to_OMEGA_in_thread)
        thread.start()

    def save_to_shelve(self, filename, info):
        with self.shelve_lock:
            with shelve.open(filename) as db:
                queue = db.get('queue', [])  # Varsayılan olarak boş liste al
                queue.append(info)
                db['queue'] = queue  # Güncellenmiş listeyi tekrar shelve'e yaz
    
    def load_from_shelve(self, filename):
        with self.shelve_lock:
            with shelve.open(filename) as db:
                return db.get('queue', [])  # Eğer 'queue' yoksa boş liste döndür

    def failed_post_to_OMEGA(self):
        def post_failed_face_result_to_OMEGA_in_thread():
            if check_connection(url='https://www.google.com'):
                failed_queue = self.load_from_shelve('failed_post_queue')
                new_queue = []  # Başarısız olanları tutacak yeni liste
                
                for failed_info in failed_queue:
                    failed_info.omega_face_post = read_json("facefinder/config/omega_face_post.json")
                    
                    try:
                        failed_info.send_post_to_db(failed_info)
                        print("Successfully sent to OMEGA, removing from queue.")
                    except Exception as e:
                        print(f"failed_post_OMEGA: {e}")
                        new_queue.append(failed_info)  # Başarısız olanları tekrar ekle
                        print("Can't push to OMEGA. Info obj is added back to failed_post_queue")

                # Güncellenmiş listeyi tekrar shelve'e yaz
                with self.shelve_lock:
                    with shelve.open('failed_post_queue', writeback=True) as db:
                        db['queue'] = new_queue  # Yeni başarısız listeyle güncelle
                        
        thread = Thread(target=post_failed_face_result_to_OMEGA_in_thread)
        thread.start()


    def load_from_shelve(self, filename):
        with self.shelve_lock:
            try:
                with shelve.open(filename) as db:
                    return db.get('queue', [])  # Eğer 'queue' yoksa boş liste döndür
            except Exception as e:
                print(f"Error loading shelve: {e}")
                return []

    def update_byteTracker(self, bbox_biggest_xyxy):
        detections = Detections(xyxy=array([bbox_biggest_xyxy]), confidence=array([0.80]), class_id=array([1]))
        detections = self.byte_tracker.update_with_detections(detections)
        return detections

    def is_same_face_within_5_sec(self, dlib_result):
        full_name, printed_name  = self.get_fullName_and_printedName(dlib_result["dlib_first_min_key"])   
        if not printed_name in self.found_face_dict.keys():
            self.found_face_dict[printed_name] = time() - 6
        if time() - self.found_face_dict[printed_name] >= 5:
            return True, full_name, printed_name
        else:
            return False, None, None

    def update_encoding_in_json(self, full_name, encodingVector, model_name, lock):
        with lock:
            # JSON dosyasını yükle
            json_path = f'{self.config_known_faces["JSON_ROOT_PATH"]}/station_{self.station_id}/{model_name.lower()}_known_faces.json'
        
            with open(json_path, 'r') as file:
                data = json.load(file)

            new_encoding_list = encodingVector.flatten().tolist()

            # İsimle eşleşen satırı bul
            match_found = False
            for entry in data:
                if entry['name'] == full_name:
                    match_found = True
                    # encodings sütunundaki mevcut veriyi al ve NumPy array'e çevir
                    existing_encoding = entry['encodings']
                    if len(existing_encoding) >= self.config_known_faces["MAX_IMAGES_PER_USER"]:
                        existing_encoding.pop(0)
                        existing_encoding.append(new_encoding_list)
                    else:
                        existing_encoding.append(new_encoding_list)

                    entry['encodings'] = existing_encoding
                    print(f"'{full_name}' için encoding güncellendi.")
                    break

            if not match_found:
                print(f"Hata: '{full_name}' için eşleşen satır bulunamadı.")

            # Güncellenmiş veriyi JSON dosyasına kaydet
            with open(json_path, 'w') as file:
                json.dump(data, file, indent=4)
                print("JSON dosyası başarıyla güncellendi.")

def count_images_in_folder(folder_path):
    try:
        # Define common image file extensions
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
        image_count = 0

        # Iterate through files in the folder
        for file_name in listdir(folder_path):
            # Check if the file extension matches common image formats
            if splitext(file_name.lower())[1] in image_extensions:
                image_count += 1
        return image_count
    except:
        return 1

def add_entry_to_json(encodedVector, model_name, station_id, lock, json_root_path):
    with lock:
        encoding = [encodedVector.flatten().tolist()]
        # Read the current data to get the last index
        data = []
        with open(f'{json_root_path}/station_{station_id}/{model_name}_known_faces.json', 'r') as file:
            data = json.load(file)
            last_index = int(data[-1]["index"]) if data else -1
        
        # Find the highest "unknown_X" index
        max_unknown_index = 0
        for row in data:
            _match = match(r"unknown_(\d+)", row.get("name", ""))
            if _match:
                max_unknown_index = max(max_unknown_index, int(_match.group(1)))

        name = f"unknown_{max_unknown_index + 1}"
        # Prepare the new entry
        new_entry = {
            "index": last_index + 1,
            "name": name,
            "encodings": encoding,
            "card_uids": ""
        }
        
        # Append the new entry to the JSON
        with open(f'{json_root_path}/station_{station_id}/{model_name}_known_faces.json', 'w') as file:
            if not data:
                json.dump([new_entry], file, indent=4)
            else:
                data.append(new_entry)
                json.dump(data, file, indent=4)
        return name