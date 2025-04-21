from multiprocessing import Queue
from os import getpid
import platform
from numpy import array
from numpy import zeros, uint8, array, ascontiguousarray, argmin
from face_recognition import face_encodings, face_distance
from deepface.DeepFace import represent
from json import load
from services.json_utils import read_json
from psutil import Process as psutil_process

class face_recognizer:
    def __init__(self, face_recognizer_queue_dict, json_lock):
        self.json_lock = json_lock
        self.known_people_dict = {}
        self.face_recognizer_queue_dict = face_recognizer_queue_dict
        self.faceID_to_faceRecognizer_queue = Queue(maxsize=1)

        self.config_known_faces = read_json("facefinder/config/known_faces.json")
        for station_id in face_recognizer_queue_dict.keys():
            self.known_people_dict[f"{station_id}_dlib"] = {}
            self.known_people_dict[f"{station_id}_dlib"]["known_face_indexes"], self.known_people_dict[f"{station_id}_dlib"]["known_face_names"], self.known_people_dict[f"{station_id}_dlib"]["known_face_encodings"], self.known_people_dict[f"{station_id}_dlib"]["known_card_uids"] = read_known_people_from_json_file(f'{self.config_known_faces["JSON_ROOT_PATH"]}/station_{station_id}/dlib_known_faces.json', self.json_lock)
            self.known_people_dict[f"{station_id}_facenet"] = {}
            self.known_people_dict[f"{station_id}_facenet"]["known_face_indexes"], self.known_people_dict[f"{station_id}_facenet"]["known_face_names"], self.known_people_dict[f"{station_id}_facenet"]["known_face_encodings"], self.known_people_dict[f"{station_id}_facenet"]["known_card_uids"] = read_known_people_from_json_file(f'{self.config_known_faces["JSON_ROOT_PATH"]}/station_{station_id}/facenet_known_faces.json', self.json_lock)
    
    def get_face_encoding(self, face_img, model_name):
        if model_name == "dlib":
            height, width = face_img.shape[:2]  # shape returns (height, width, channels)
            # Define the bounding box in (top, right, bottom, left) format
            top = 0
            right = width
            bottom = height
            left = 0
            bounding_box = (top, right, bottom, left)
            dlib_encodedVector = array(face_encodings(ascontiguousarray(face_img[:, :, ::-1]), [bounding_box]))
            return dlib_encodedVector
        elif model_name == "facenet":
            img_objs = represent(img_path=face_img, detector_backend="skip", model_name="Facenet512")
            facenet_encodedVector = array([img_objs[0]["embedding"]])
            return facenet_encodedVector

    def compare_face_encodings(self, encodedVector, model_name, station_id):
        face_distances_all = {}
        min_key = ""
        min_val = 1
        second_min_key = ""
        second_min_val = 1
        if model_name == "dlib":
            known_face_encodings = self.known_people_dict[f"{station_id}_dlib"]["known_face_encodings"]
            known_face_names = self.known_people_dict[f"{station_id}_dlib"]["known_face_names"]
        elif model_name == "facenet":
            known_face_encodings = self.known_people_dict[f"{station_id}_facenet"]["known_face_encodings"]
            known_face_names = self.known_people_dict[f"{station_id}_facenet"]["known_face_names"]

        else:
            print("Model name uygun değil. -compare_face_encodings")
            return

        for i, known_face_encodings_per_person in enumerate(known_face_encodings):
            if known_face_encodings_per_person != []:
                try:
                    for encoding_known_face_encodings_per_person in known_face_encodings_per_person:
                        encoding_known_face_encodings_per_person = array(encoding_known_face_encodings_per_person)
                        face_distances = face_distance(encoding_known_face_encodings_per_person, encodedVector)
                except Exception as e:
                    print(f"problem: {e}")
                # Get the name of the best match
                best_match_index = argmin(face_distances)
                face_distances_all[known_face_names[i]] = face_distances[best_match_index]

        # Find the minimum and second minimum distances
        sorted_distances = sorted(face_distances_all.items(), key=lambda x: x[1])
        if len(sorted_distances) > 0:
            min_key, min_val = sorted_distances[0]
        if len(sorted_distances) > 1:
            second_min_key, second_min_val = sorted_distances[1]

        if min_val <= 0.55:
            print(f"Best Match: {min_val}, {min_key}, {station_id}")
            print(f"Second Best Match: {second_min_val}, {second_min_key}, {station_id}")
        result = {}
        if model_name == "dlib":
            result["dlib_first_min_key"] = min_key
            result["dlib_first_min_val"] = round(min_val, 2)
            result["dlib_second_min_key"] = second_min_key
            result["dlib_second_min_val"] = round(second_min_val, 2)
        elif model_name == "facenet":
            result["facenet_first_min_key"] = min_key
            result["facenet_first_min_val"] = round(min_val, 2)
            result["facenet_second_min_key"] = second_min_key
            result["facenet_second_min_val"] = round(second_min_val, 2)
        return result
    
    def get_face_id_result(self, face_img, model_name, station_id):
        encodedVector = self.get_face_encoding(face_img, model_name)
        result = self.compare_face_encodings(encodedVector, model_name, station_id)
        return encodedVector, result

    def warmup_models(self):
        dummy_img = zeros((150,150,3), dtype=uint8)
        dummy_encodingVector = self.get_face_encoding(dummy_img, "dlib")
        dummy_encodingVector = self.get_face_encoding(dummy_img, "facenet")

    def face_recognizer_MP(self, stop_event, _):
            system = platform.system()
            if system == "Linux":
                process = psutil_process(getpid())
                process.cpu_affinity([2])
            self.warmup_models()

            while not stop_event.is_set():
                face_img, model_name, station_id = self.faceID_to_faceRecognizer_queue.get(block=True)
                if face_img is None and model_name is None:
                    self.read_csv_files(station_id)
                    self.config_known_faces = read_json("facefinder/config/known_faces.json")
                else:
                    print(f"faceRecognizer is processing frames for {station_id}")
                    encodedVector, result = self.get_face_id_result(face_img, model_name, station_id)
                    self.face_recognizer_queue_dict[str(station_id)]["faceRecognizer_to_faceID_queue"].put([encodedVector, result])

    def read_csv_files(self, station_id):
        self.known_people_dict[f"{station_id}_dlib"] = {}
        self.known_people_dict[f"{station_id}_dlib"]["known_face_indexes"], self.known_people_dict[f"{station_id}_dlib"]["known_face_names"], self.known_people_dict[f"{station_id}_dlib"]["known_face_encodings"], self.known_people_dict[f"{station_id}_dlib"]["known_card_uids"] = read_known_people_from_json_file(f'{self.config_known_faces["JSON_ROOT_PATH"]}/station_{station_id}/dlib_known_faces.json', self.json_lock)
        self.known_people_dict[f"{station_id}_facenet"] = {}
        self.known_people_dict[f"{station_id}_facenet"]["known_face_indexes"], self.known_people_dict[f"{station_id}_facenet"]["known_face_names"], self.known_people_dict[f"{station_id}_facenet"]["known_face_encodings"], self.known_people_dict[f"{station_id}_facenet"]["known_card_uids"] = read_known_people_from_json_file(f'{self.config_known_faces["JSON_ROOT_PATH"]}/station_{station_id}/facenet_known_faces.json', self.json_lock)

def read_known_people_from_json_file(file_path, lock):
    with lock:
        known_face_indexes = []
        known_face_names = []
        known_face_encodings = []
        known_card_uids = []
        with open(file_path, 'r') as jsonfile:
            data = load(jsonfile)

            for item in data:
                known_face_indexes.append(item['index'])
                known_face_names.append(item['name'])
                known_face_encodings.append(item['encodings'])
                known_card_uids.append(item["card_uids"])

        return known_face_indexes, known_face_names, known_face_encodings, known_card_uids

