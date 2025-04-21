# from csv import field_size_limit
# from sys import maxsize
# from time import time
# from numpy import zeros, uint8, array, ascontiguousarray, argmin, mean, arctan2, pi, int32, zeros_like, ones_like, where
# from face_recognition import face_encodings, face_distance
# from deepface.DeepFace import represent
# from json import load, dump, loads
# from re import match
# from global_functions.app_constants import CSV_ROOT_PATH, MAX_IMAGES_PER_USER
# from services.tracking import draw_tracking_bbox_on_frame
# from services.utils import get_biggest_bbox
# from mediapipe.python.solutions.face_mesh import FaceMesh
# from cv2 import getRotationMatrix2D, warpAffine, INTER_CUBIC, fillPoly, bitwise_and, COLOR_BGR2RGB, cvtColor, convexHull, fillConvexPoly, resize, COLOR_BGR2GRAY, findNonZero, boundingRect, createCLAHE, GaussianBlur, Canny
# from cv2 import mean as cv2_mean
# from numpy.linalg import norm
# from ultralytics import YOLO
# import dlib
# dlib.DLIB_NUM_THREADS = 4




# field_size_limit(maxsize)
# class face_recognizer:
#     def __init__(self, station_id):
#         self.station_id = station_id
#         self.dlib_csv_path = f'{CSV_ROOT_PATH}/station_{station_id}/dlib_known_faces.json'
#         self.facenet_csv_file_path = f'{CSV_ROOT_PATH}/station_{station_id}/facenet_known_faces.json'
#         self.dlib_known_face_indexes, self.dlib_known_face_names, self.dlib_known_face_encodings, self.dlib_known_card_uids = read_known_people_from_json_file(self.dlib_csv_path)
#         self.facenet_known_face_indexes, self.facenet_known_face_names, self.facenet_known_face_encodings, self.facenet_known_card_uids = read_known_people_from_json_file(self.facenet_csv_file_path)

#     def get_face_encoding(self, face_img, model_name):
#         if model_name == "dlib":
#             height, width = face_img.shape[:2]  # shape returns (height, width, channels)
#             # Define the bounding box in (top, right, bottom, left) format
#             top = 0
#             right = width
#             bottom = height
#             left = 0
#             bounding_box = (top, right, bottom, left)
#             dlib_encodedVector = array(face_encodings(ascontiguousarray(face_img[:, :, ::-1]), [bounding_box]))
#             return dlib_encodedVector
#         elif model_name == "facenet":
#             img_objs = represent(img_path=face_img, detector_backend="skip", model_name="Facenet512")
#             facenet_encodedVector = array([img_objs[0]["embedding"]])
#             return facenet_encodedVector

#     def compare_face_encodings(self, encodedVector, model_name):
#         face_distances_all = {}
#         min_key = ""
#         min_val = 1
#         second_min_key = ""
#         second_min_val = 1
#         if model_name == "dlib":
#             known_face_encodings = self.dlib_known_face_encodings
#             known_face_names = self.dlib_known_face_names
#         elif model_name == "facenet":
#             known_face_encodings = self.facenet_known_face_encodings
#             known_face_names = self.facenet_known_face_names

#         else:
#             print("Model name uygun değil. -compare_face_encodings")
#             return

#         for i, known_face_encodings_per_person in enumerate(known_face_encodings):
#             if known_face_encodings_per_person != []:
#                 try:
#                     for encoding_known_face_encodings_per_person in known_face_encodings_per_person:
#                         encoding_known_face_encodings_per_person = array(encoding_known_face_encodings_per_person)
#                         face_distances = face_distance(encoding_known_face_encodings_per_person, encodedVector)
#                 except Exception as e:
#                     print(f"problem: {e}")
#                 # Get the name of the best match
#                 best_match_index = argmin(face_distances)
#                 face_distances_all[known_face_names[i]] = face_distances[best_match_index]

#         # Find the minimum and second minimum distances
#         sorted_distances = sorted(face_distances_all.items(), key=lambda x: x[1])
#         if len(sorted_distances) > 0:
#             min_key, min_val = sorted_distances[0]
#         if len(sorted_distances) > 1:
#             second_min_key, second_min_val = sorted_distances[1]

#         if min_val <= 0.55:
#             print(f"Best Match: {min_val}, {min_key}, {self.station_id}")
#             print(f"Second Best Match: {second_min_val}, {second_min_key}, {self.station_id}")
#         result = {}
#         if model_name == "dlib":
#             result["dlib_first_min_key"] = min_key
#             result["dlib_first_min_val"] = round(min_val, 2)
#             result["dlib_second_min_key"] = second_min_key
#             result["dlib_second_min_val"] = round(second_min_val, 2)
#         elif model_name == "facenet":
#             result["facenet_first_min_key"] = min_key
#             result["facenet_first_min_val"] = round(min_val, 2)
#             result["facenet_second_min_key"] = second_min_key
#             result["facenet_second_min_val"] = round(second_min_val, 2)
#         return result
    
#     def get_face_id_result(self, face_img, model_name):
#         encodedVector = self.get_face_encoding(face_img, model_name)
#         result = self.compare_face_encodings(encodedVector, model_name)
#         return encodedVector, result

#     def warmup_models(self):
#         dummy_img = zeros((150,150,3), dtype=uint8)
#         dummy_encodingVector = self.get_face_encoding(dummy_img, "dlib")
#         dummy_encodingVector = self.get_face_encoding(dummy_img, "facenet")

#     def read_csv_files(self):
#         self.dlib_known_face_indexes, self.dlib_known_face_names, self.dlib_known_face_encodings, self.dlib_known_card_uids = read_known_people_from_json_file(self.dlib_csv_path)
#         self.facenet_known_face_indexes, self.facenet_known_face_names, self.facenet_known_face_encodings, self.facenet_known_card_uids = read_known_people_from_json_file(self.facenet_csv_file_path)


#     def add_entry_to_json(self, encodedVector, model_name):
#         encoding = [encodedVector.flatten().tolist()]
#         # Read the current data to get the last index
#         data = []
#         with open(f'{CSV_ROOT_PATH}/station_{self.station_id}/{model_name}_known_faces.json', 'r') as file:
#             data = load(file)
#             last_index = int(data[-1]["index"]) if data else -1
        
#         # Find the highest "unknown_X" index
#         max_unknown_index = 0
#         for row in data:
#             _match = match(r"unknown_(\d+)", row.get("name", ""))
#             if _match:
#                 max_unknown_index = max(max_unknown_index, int(_match.group(1)))

#         name = f"unknown_{max_unknown_index + 1}"
#         # Prepare the new entry
#         new_entry = {
#             "index": last_index + 1,
#             "name": name,
#             "encodings": encoding,
#             "card_uids": ""
#         }
        
#         # Append the new entry to the JSON
#         with open(f'{CSV_ROOT_PATH}/station_{self.station_id}/{model_name}_known_faces.json', 'w') as file:
#             if not data:
#                 dump([new_entry], file, indent=4)
#             else:
#                 data.append(new_entry)
#                 dump(data, file, indent=4)
#         return name

# def parse_encoded_images(encoded_image_str):
#     # JSON formatındaki veriyi tekrar numpy array formatına çevir
#     return array(loads(encoded_image_str))

# def read_known_people_from_json_file(file_path):
#     known_face_indexes = []
#     known_face_names = []
#     known_face_encodings = []
#     known_card_uids = []

#     with open(file_path, 'r') as jsonfile:
#         data = load(jsonfile)

#         for item in data:
#             known_face_indexes.append(item['index'])
#             known_face_names.append(item['name'])
#             known_face_encodings.append(item['encodings'])
#             known_card_uids.append(item["card_uids"])

#     return known_face_indexes, known_face_names, known_face_encodings, known_card_uids

# class landmark_detector:
#     def __init__(self,
#                  usage_reason,         
#                 up_threshold = 0.5,  # Adjust this threshold
#                 down_threshold = 2,  # Adjust this threshold
#                 left_threshold = 10,  # Adjust this threshold
#                 right_threshold = 0.2,
#                 display_mode=True,
#                 station=None):
#         self.station = station
#         self.display_mode = display_mode
#         self.usage_reason = usage_reason
#         if usage_reason == "realtime":
#             self.face_mesh = FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.88, min_tracking_confidence=0.95)
#         if usage_reason == "image":
#             self.face_mesh = FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.70)

#         self.up_threshold = up_threshold  # Adjust this threshold
#         self.down_threshold = down_threshold  # Adjust this threshold
#         self.left_threshold = left_threshold  # Adjust this threshold
#         self.right_threshold = right_threshold # Adjust this threshold

#         self.landmarks_prev = None
#         self.landmarks_curr = None
#         self.movements = []

#     def align_face(self, image, landmarks):
#         img_h, img_w, _ = image.shape

#         # Define eye landmarks (left eye and right eye)
#         left_eye = [landmarks[33], landmarks[133]]  # Example landmark indices for eyes
#         right_eye = [landmarks[362], landmarks[263]]

#         # Calculate the center of eyes
#         left_eye_center = mean(left_eye, axis=0)
#         right_eye_center = mean(right_eye, axis=0)

#         # Calculate the angle between the eyes
#         dx = right_eye_center[0] - left_eye_center[0]
#         dy = right_eye_center[1] - left_eye_center[1]
#         angle = arctan2(dy, dx) * 180. / pi

#         # Rotate the image around the center
#         center = (img_w / 2, img_h / 2)
#         rot_mat = getRotationMatrix2D(center, angle, 1.0)
#         aligned_image = warpAffine(image, rot_mat, (img_w, img_h), flags=INTER_CUBIC)

#         return aligned_image

#     def check_head_yaw_movement(self, landmarks, img_w, img_h):
#         # Select landmark points (e.g., left eye, right eye, nose, and ears)
#         left_eye_idx = 33  # Left eye outer corner
#         right_eye_idx = 263  # Right eye outer corner
#         nose_idx = 1  # Nose tip
#         left_ear_idx = 234  # Left ear
#         right_ear_idx = 454  # Right ear

#         # Get the 3D coordinates of these landmarks
#         left_eye = landmarks[left_eye_idx]
#         right_eye = landmarks[right_eye_idx]
#         nose = landmarks[nose_idx]
#         left_ear = landmarks[left_ear_idx]
#         right_ear = landmarks[right_ear_idx]

#         # Calculate the face width based on the ear landmarks
#         face_width = abs(left_ear.x - right_ear.x) * img_w

#         # Calculate the horizontal distance between the nose and the eyes
#         left_eye_to_nose_dist = abs(nose.x - left_eye.x) * img_w
#         right_eye_to_nose_dist = abs(nose.x - right_eye.x) * img_w

#         # Normalize the distances by the face width
#         normalized_left_eye_to_nose = left_eye_to_nose_dist / face_width
#         normalized_right_eye_to_nose = right_eye_to_nose_dist / face_width

#         # Calculate the ratio between the normalized distances
#         ratio = normalized_left_eye_to_nose / normalized_right_eye_to_nose if normalized_right_eye_to_nose != 0 else 0
#         # Determine the head movement direction
#         if normalized_right_eye_to_nose > 0.50:
#             movement = "Turned Right"
#         elif normalized_left_eye_to_nose > 0.50:
#             movement = "Turned Left"
#         else:
#             movement = "Facing Forward"
#         return movement, ratio, left_eye, right_eye, nose

#     def check_head_pitch_movement(self, landmarks, img_w, img_h):
#         # Select landmark points
#         nose_idx = 1  # Nose tip
#         chin_idx = 152  # Chin
#         forehead_idx = 10  # Forehead (between the eyes)

#         # Get the 3D coordinates of these landmarks
#         nose = landmarks[nose_idx]
#         chin = landmarks[chin_idx]
#         forehead = landmarks[forehead_idx]

#         # Calculate face height based on the forehead and chin landmarks
#         face_height = abs(forehead.y - chin.y) * img_h

#         # Calculate the vertical distance between the nose and the chin
#         nose_to_chin_dist = abs(nose.y - chin.y) * img_h

#         # Normalize the distance by the face height
#         normalized_nose_to_chin = nose_to_chin_dist / face_height

#         # Calculate the vertical distance between the nose and the forehead
#         nose_to_forehead_dist = abs(nose.y - forehead.y) * img_h

#         # Normalize the distance by the face height
#         normalized_nose_to_forehead = nose_to_forehead_dist / face_height

#         # Calculate the ratio between the two normalized distances
#         ratio = normalized_nose_to_forehead / normalized_nose_to_chin if normalized_nose_to_chin != 0 else 0

#         # Determine the head tilt direction (bottom-to-top movement)
#         if normalized_nose_to_forehead > 0.65:
#             movement = "Tilted Up"
#         elif normalized_nose_to_chin > 0.65:
#             movement = "Tilted Down"
#         else:
#             movement = "Facing Forward"
#         return movement, ratio, nose, chin, forehead

#     def get_color_distribution(self, region, image):
#         # Belirli bir bölgenin renk dağılımını hesaplayın
#         mask = zeros(image.shape[:2], dtype=uint8)
#         fillPoly(mask, [region], (255))
        
#         masked_image = bitwise_and(image, image, mask=mask)
#         average_color = cv2_mean(masked_image, mask=mask)[:3]  # Renk ortalamasını al
#         return average_color
    
#     def crop_face_from_landmarks(self, image):
#         # Convert the image to RGB
#         rgb_image = cvtColor(image, COLOR_BGR2RGB)
#         # Perform face landmark detection
#         results = self.face_mesh.process(rgb_image)
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 self.landmarks_curr = face_landmarks.landmark
#                 points = []
#                 img_h, img_w, _ = image.shape
#                 for landmark in face_landmarks.landmark:
#                     x = int(landmark.x * img_w)
#                     y = int(landmark.y * img_h)
#                     points.append((x, y))

#                 # Convert points to a numpy array
#                 points_np = array(points, int32)

#                 # Create a mask
#                 mask = zeros_like(image)
#                 # Define the polygon for the face using the landmarks
#                 hull = convexHull(points_np)
#                 # Fill the mask with the face region
#                 fillConvexPoly(mask, hull, (255, 255, 255))

#                 # Apply the mask to the original image
#                 cropped_face = bitwise_and(image, mask)
#                 # Set the background to black
#                 background = ones_like(image) * 0  # Black background
#                 masked_face = where(mask == (255, 255, 255), cropped_face, background)
#                 # Align the face based on eyes
#                 aligned_face = self.align_face(masked_face, points_np)
#                 # Crop from the borders of the face
#                 if len(points_np) > 0:
#                     x_min = min([x for x, y in points_np])
#                     x_max = max([x for x, y in points_np])
#                     y_min = min([y for x, y in points_np])
#                     y_max = max([y for x, y in points_np])
#                 else:
#                     return image, None, None, 0, "", ""

#                 # Add a buffer to the cropped area if needed (optional)
#                 buffer = 0
#                 x_min = max(x_min - buffer, 0)
#                 x_max = min(x_max + buffer, aligned_face.shape[1])
#                 y_min = max(y_min - buffer, 0)
#                 y_max = min(y_max + buffer, aligned_face.shape[0])
#                 bboxes = [x_min, y_min, x_max, y_max]
#                 boxes = array([[bboxes[0], bboxes[1], bboxes[2], bboxes[3]]])
#                 bbox_biggest_xyxy = get_biggest_bbox(boxes)
#                 draw_tracking_bbox_on_frame(image, bbox_biggest_xyxy, bbox_type="xyxy", moving_average=False)
#                 is_head_moving, mov_mag = self.is_head_moving()
#                 if not is_head_moving or self.usage_reason == "image":
#                     # Collect the landmark points
#                     movement_yaw, yaw_ratio, left_eye, right_eye, nose = self.check_head_yaw_movement(self.landmarks_curr, img_w, img_h)
#                     movement_pitch, pitch_ratio, nose, chin, forehead = self.check_head_pitch_movement(self.landmarks_curr, img_w, img_h)
#                     if movement_yaw == "Facing Forward" and movement_pitch == "Facing Forward":
#                         if self.is_face_fully_in_frame():
#                             cropped_face = crop_face_based_on_masked_image(aligned_face)
#                             cropped_face_final = resize(cropped_face, (150, 150))
#                             return image, cropped_face_final, bbox_biggest_xyxy, 0, "", "", mov_mag

#                         else:
#                             #print("Face is not fully on display. Skipping...")
#                             return image, None, None, 1, "Cok yakinsiniz", "Biraz uzaklasin", None
#                     else:
#                         #print("Face is not frontal. Skipping...")
#                         return image, None, None, 1, "Yuz yan duruyor", "Kameraya bakin", None
#                 else:
#                     #print("Head is moving. Skipping...")
#                     return image, None, None, 1, "Yuz sabit degil", "Hareket etmeyin", None
#         else:
#             return image, None, None, 0, "", "", None

#     def calculate_movement_magnitude(self):
#         if self.landmarks_prev is None or self.landmarks_curr is None:
#             return 0
#         # Burun landmark indeksini belirliyoruz
#         nose_index = 1  # Mediapipe modeline göre doğrulayın
#         prev = self.landmarks_prev[nose_index]
#         curr = self.landmarks_curr[nose_index]
        
#         # Hareket büyüklüğünü hesaplıyoruz
#         movement = norm(array([curr.x - prev.x, curr.y - prev.y]))
    
#         return movement

#     def is_head_moving(self):
#         movement_magnitude = self.calculate_movement_magnitude()
#         self.landmarks_prev = self.landmarks_curr
#         if self.usage_reason == "realtime":
#             if self.station.station_id == 2:
#                 if movement_magnitude < 0.03:
#                     return False, movement_magnitude
#                 else:
#                     return True, movement_magnitude
#             else:
#                 if movement_magnitude < 0.008:
#                     return False, movement_magnitude
#                 else:
#                     return True, movement_magnitude
#         else:
#             if movement_magnitude < 0.03:
#                 return False, movement_magnitude
#             else:
#                 return True, movement_magnitude
            
#     def is_face_fully_in_frame(self):
#         for landmark in self.landmarks_curr:
#             x, y = landmark.x, landmark.y
#             # Check if landmark is within the frame boundaries
#             if not (0 <= x <= 1 and 0 <= y <= 1):
#                 return False  # Part of the face is out of the frame
#         return True  # All landmarks are within the frame
    
# def crop_face_based_on_masked_image(masked_image):
#     # Convert the image to grayscale
#     gray_image = cvtColor(masked_image, COLOR_BGR2GRAY)

#     # Find the coordinates of non-black (colored) pixels
#     coords = findNonZero(gray_image)

#     if coords is not None:
#         # Get the bounding box of the non-black pixels
#         x_min, y_min, w, h = boundingRect(coords)
#         x_max = x_min + w
#         y_max = y_min + h

#         # Crop the masked image using these boundaries
#         cropped_face = masked_image[y_min:y_max, x_min:x_max]

#         return cropped_face
#     else:
#         print("No colored pixels found")
#         return None

# class PPE_checker:
#     def __init__(self):
#         self.model_path = "facefinder/models/ppe_gray_v11_ncnn_model"
#         self.model = YOLO(self.model_path, task="detect")

#     def predict_PPE(self, person_img):
#         st = time()
#         results = self.model(person_img, conf=0.25, verbose=False)
#         et = time()
#         print(et-st)
#         is_helmet = False
#         # Draw detections on the image
#         for i, result in enumerate(results[0].boxes):
#             if result.cls == 10:
#                 is_helmet = True
#         return is_helmet

# def obstacle_checker(frame, face_landmarks):
#     ih, iw, _ = frame.shape
#     # Landmarks for the left and right outer eyes
#     left_eye_outer_top_index = 107   # Left side of the left eye
#     left_eye_outer_bottom_index = 196
#     right_eye_outer_top_index = 336  # Right side of the right eye
#     right_eye_outer_bottom_index = 419
#     # Get the coordinates of the outermost points of both eyes
#     left_eye_outer_top = [face_landmarks.landmark[left_eye_outer_top_index].x, face_landmarks.landmark[left_eye_outer_top_index].y]
#     left_eye_outer_bottom = [face_landmarks.landmark[left_eye_outer_bottom_index].x, face_landmarks.landmark[left_eye_outer_bottom_index].y]

#     right_eye_outer_top = [face_landmarks.landmark[right_eye_outer_top_index].x, face_landmarks.landmark[right_eye_outer_top_index].y]
#     right_eye_outer_bottom = [face_landmarks.landmark[right_eye_outer_bottom_index].x, face_landmarks.landmark[right_eye_outer_bottom_index].y]

#     x_min = int(min(left_eye_outer_top[0], left_eye_outer_bottom[0])* iw)
#     x_max = int(max(right_eye_outer_top[0], right_eye_outer_bottom[0]) * iw)
#     y_min = int(min(left_eye_outer_top[1], right_eye_outer_top[1]) * ih)
#     y_max = int(max(left_eye_outer_bottom[1], right_eye_outer_bottom[1])* ih)

#     # Crop the region between the two eyes
#     eye_region = frame[y_min:y_max, int(x_min+(x_max-x_min)/3):int(x_max-(x_max-x_min)/3)]

#     # Display the cropped region
#     eye_region = cvtColor(eye_region, COLOR_BGR2GRAY)
#     # gray_eye_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
#     clahe = createCLAHE(clipLimit=2, tileGridSize=(2, 2))
#     eye_region = clahe.apply(eye_region)
#     img_blur = GaussianBlur(array(eye_region),(5,5), sigmaX=1.7, sigmaY=1.7)
#     edges = Canny(image=img_blur, threshold1=30, threshold2=150)
#     # Optionally, draw a rectangle around the eyes on the original frame
#     # Apply dilation to thicken the edges
#     # Optionally apply erosion after dilation
#     edges_center = edges.T[(int(len(edges.T)/2))]
#     edges_left = edges.T[(int(len(edges.T)/10))]
#     edges_left2 = edges.T[(int(3*len(edges.T)/10))]
#     edges_right2 = edges.T[(int(7*len(edges.T)/10))]
#     edges_right = edges.T[(int(9*len(edges.T)/10))]
#     # Check for the presence of white edges indicating glasses
#     if 255 in edges_center or 255 in edges_left or 255 in edges_left2 or 255 in edges_right or 255 in edges_right2:
#         # Display message indicating presence of glasses
#         return "Glass is Present"
#     else:
#         # Display message indicating absence of glasses
#         return "No Glass"