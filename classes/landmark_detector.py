from multiprocessing import Queue
from threading import Thread
from numpy import zeros, uint8, array, mean, arctan2, pi, int32, zeros_like, ones_like, where
from services.tracking import draw_tracking_bbox_on_frame
from services.utils import get_biggest_bbox
from mediapipe.python.solutions.face_mesh import FaceMesh
from cv2 import COLOR_BGR2GRAY, boundingRect, findNonZero, getRotationMatrix2D, warpAffine, INTER_CUBIC, fillPoly, bitwise_and, COLOR_BGR2RGB, cvtColor, convexHull, fillConvexPoly, resize
from cv2 import mean as cv2_mean
from numpy.linalg import norm

class landmark_detector:
    def __init__(self,
                 usage_reason, 
                 landmark_queue_dict, 
                 station_id,       
                up_threshold = 0.5,  # Adjust this threshold
                down_threshold = 2,  # Adjust this threshold
                left_threshold = 10,  # Adjust this threshold
                right_threshold = 0.2,
                display_mode=True):
        self.display_mode = display_mode
        self.usage_reason = usage_reason
        self.station_id = station_id
        if usage_reason == "realtime":
            self.face_mesh = FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.88, min_tracking_confidence=0.95)
        if usage_reason == "image":
            self.face_mesh = FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.70)

        self.up_threshold = up_threshold  # Adjust this threshold
        self.down_threshold = down_threshold  # Adjust this threshold
        self.left_threshold = left_threshold  # Adjust this threshold
        self.right_threshold = right_threshold # Adjust this threshold

        self.landmarks_prev = None
        self.landmarks_curr = None
        self.movements = []
        self.landmark_queue_dict = landmark_queue_dict
        self.faceDet_to_landmark_queue = Queue(maxsize=1)

    def align_face(self, image, landmarks):
        img_h, img_w, _ = image.shape

        # Define eye landmarks (left eye and right eye)
        left_eye = [landmarks[33], landmarks[133]]  # Example landmark indices for eyes
        right_eye = [landmarks[362], landmarks[263]]

        # Calculate the center of eyes
        left_eye_center = mean(left_eye, axis=0)
        right_eye_center = mean(right_eye, axis=0)

        # Calculate the angle between the eyes
        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        angle = arctan2(dy, dx) * 180. / pi

        # Rotate the image around the center
        center = (img_w / 2, img_h / 2)
        rot_mat = getRotationMatrix2D(center, angle, 1.0)
        aligned_image = warpAffine(image, rot_mat, (img_w, img_h), flags=INTER_CUBIC)

        return aligned_image

    def check_head_yaw_movement(self, landmarks, img_w, img_h):
        # Select landmark points (e.g., left eye, right eye, nose, and ears)
        left_eye_idx = 33  # Left eye outer corner
        right_eye_idx = 263  # Right eye outer corner
        nose_idx = 1  # Nose tip
        left_ear_idx = 234  # Left ear
        right_ear_idx = 454  # Right ear

        # Get the 3D coordinates of these landmarks
        left_eye = landmarks[left_eye_idx]
        right_eye = landmarks[right_eye_idx]
        nose = landmarks[nose_idx]
        left_ear = landmarks[left_ear_idx]
        right_ear = landmarks[right_ear_idx]

        # Calculate the face width based on the ear landmarks
        face_width = abs(left_ear.x - right_ear.x) * img_w

        # Calculate the horizontal distance between the nose and the eyes
        left_eye_to_nose_dist = abs(nose.x - left_eye.x) * img_w
        right_eye_to_nose_dist = abs(nose.x - right_eye.x) * img_w

        # Normalize the distances by the face width
        normalized_left_eye_to_nose = left_eye_to_nose_dist / face_width
        normalized_right_eye_to_nose = right_eye_to_nose_dist / face_width

        # Calculate the ratio between the normalized distances
        ratio = normalized_left_eye_to_nose / normalized_right_eye_to_nose if normalized_right_eye_to_nose != 0 else 0
        # Determine the head movement direction
        if normalized_right_eye_to_nose > 0.50:
            movement = "Turned Right"
        elif normalized_left_eye_to_nose > 0.50:
            movement = "Turned Left"
        else:
            movement = "Facing Forward"
        return movement, ratio, left_eye, right_eye, nose

    def check_head_pitch_movement(self, landmarks, img_w, img_h):
        # Select landmark points
        nose_idx = 1  # Nose tip
        chin_idx = 152  # Chin
        forehead_idx = 10  # Forehead (between the eyes)

        # Get the 3D coordinates of these landmarks
        nose = landmarks[nose_idx]
        chin = landmarks[chin_idx]
        forehead = landmarks[forehead_idx]

        # Calculate face height based on the forehead and chin landmarks
        face_height = abs(forehead.y - chin.y) * img_h

        # Calculate the vertical distance between the nose and the chin
        nose_to_chin_dist = abs(nose.y - chin.y) * img_h

        # Normalize the distance by the face height
        normalized_nose_to_chin = nose_to_chin_dist / face_height

        # Calculate the vertical distance between the nose and the forehead
        nose_to_forehead_dist = abs(nose.y - forehead.y) * img_h

        # Normalize the distance by the face height
        normalized_nose_to_forehead = nose_to_forehead_dist / face_height

        # Calculate the ratio between the two normalized distances
        ratio = normalized_nose_to_forehead / normalized_nose_to_chin if normalized_nose_to_chin != 0 else 0

        # Determine the head tilt direction (bottom-to-top movement)
        if normalized_nose_to_forehead > 0.65:
            movement = "Tilted Up"
        elif normalized_nose_to_chin > 0.65:
            movement = "Tilted Down"
        else:
            movement = "Facing Forward"
        return movement, ratio, nose, chin, forehead

    def get_color_distribution(self, region, image):
        # Belirli bir bölgenin renk dağılımını hesaplayın
        mask = zeros(image.shape[:2], dtype=uint8)
        fillPoly(mask, [region], (255))
        
        masked_image = bitwise_and(image, image, mask=mask)
        average_color = cv2_mean(masked_image, mask=mask)[:3]  # Renk ortalamasını al
        return average_color
    
    def crop_face_from_landmarks(self, image):
        rgb_image = cvtColor(image, COLOR_BGR2RGB)
        # Perform face landmark detection
        results = self.face_mesh.process(rgb_image)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.landmarks_curr = face_landmarks.landmark
                points = []
                img_h, img_w, _ = image.shape
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * img_w)
                    y = int(landmark.y * img_h)
                    points.append((x, y))

                # Convert points to a numpy array
                points_np = array(points, int32)

                # Create a mask
                mask = zeros_like(image)
                # Define the polygon for the face using the landmarks
                hull = convexHull(points_np)
                # Fill the mask with the face region
                fillConvexPoly(mask, hull, (255, 255, 255))

                # Apply the mask to the original image
                cropped_face = bitwise_and(image, mask)
                # Set the background to black
                background = ones_like(image) * 0  # Black background
                masked_face = where(mask == (255, 255, 255), cropped_face, background)
                # Align the face based on eyes
                aligned_face = self.align_face(masked_face, points_np)
                # Crop from the borders of the face
                if len(points_np) > 0:
                    x_min = min([x for x, y in points_np])
                    x_max = max([x for x, y in points_np])
                    y_min = min([y for x, y in points_np])
                    y_max = max([y for x, y in points_np])
                else:
                    return image, None, None, 0, "", ""

                # Add a buffer to the cropped area if needed (optional)
                buffer = 0
                x_min = max(x_min - buffer, 0)
                x_max = min(x_max + buffer, aligned_face.shape[1])
                y_min = max(y_min - buffer, 0)
                y_max = min(y_max + buffer, aligned_face.shape[0])
                bboxes = [x_min, y_min, x_max, y_max]
                boxes = array([[bboxes[0], bboxes[1], bboxes[2], bboxes[3]]])
                bbox_biggest_xyxy = get_biggest_bbox(boxes)
                #draw_tracking_bbox_on_frame(image, bbox_biggest_xyxy, bbox_type="xyxy", moving_average=False)
                is_head_moving, mov_mag = self.is_head_moving()
                if not is_head_moving or self.usage_reason == "image":
                    # Collect the landmark points
                    movement_yaw, yaw_ratio, left_eye, right_eye, nose = self.check_head_yaw_movement(self.landmarks_curr, img_w, img_h)
                    movement_pitch, pitch_ratio, nose, chin, forehead = self.check_head_pitch_movement(self.landmarks_curr, img_w, img_h)
                    if movement_yaw == "Facing Forward" and movement_pitch == "Facing Forward":
                        if self.is_face_fully_in_frame():
                            cropped_face = crop_face_based_on_masked_image(aligned_face)
                            cropped_face_final = resize(cropped_face, (150, 150))
                            return image, cropped_face_final, bbox_biggest_xyxy, 0, "", "", mov_mag

                        else:
                            #print("Face is not fully on display. Skipping...")
                            return image, None, None, 1, "Cok yakinsiniz", "Biraz uzaklasin", None
                    else:
                        #print("Face is not frontal. Skipping...")
                        return image, None, None, 1, "Yuz yan duruyor", "Kameraya bakin", None
                else:
                    #print("Head is moving. Skipping...")
                    return image, None, None, 1, "Yuz sabit degil", "Hareket etmeyin", None
        else:
            return image, None, None, 0, "", "", None

    def calculate_movement_magnitude(self):
        if self.landmarks_prev is None or self.landmarks_curr is None:
            return 0
        # Burun landmark indeksini belirliyoruz
        nose_index = 1  # Mediapipe modeline göre doğrulayın
        prev = self.landmarks_prev[nose_index]
        curr = self.landmarks_curr[nose_index]
        
        # Hareket büyüklüğünü hesaplıyoruz
        movement = norm(array([curr.x - prev.x, curr.y - prev.y]))
    
        return movement

    def is_head_moving(self):
        movement_magnitude = self.calculate_movement_magnitude()
        self.landmarks_prev = self.landmarks_curr
        if self.usage_reason == "realtime":
            if movement_magnitude < 0.008:
                return False, movement_magnitude
            else:
                return True, movement_magnitude
        else:
            if movement_magnitude < 0.03:
                return False, movement_magnitude
            else:
                return True, movement_magnitude
            
    def is_face_fully_in_frame(self):
        for landmark in self.landmarks_curr:
            x, y = landmark.x, landmark.y
            # Check if landmark is within the frame boundaries
            if not (0 <= x <= 1 and 0 <= y <= 1):
                return False  # Part of the face is out of the frame
        return True  # All landmarks are within the frame
    
    def faceLandmarkDet(self):

        def thread_faceLandmarkDet():
            while True:
                frame, station_id = self.faceDet_to_landmark_queue.get(block=True)
                print(f"faceLandmarkDet is processing frames for {station_id}")
                image, face_img, bbox_biggest_xyxy, flag_face_skip, user_warning1, user_warning2, mov_mag = self.crop_face_from_landmarks(frame)
                self.landmark_queue_dict[f"{station_id}"]["landmark_to_faceDet_queue"].put([image, face_img, bbox_biggest_xyxy, flag_face_skip, user_warning1, user_warning2, mov_mag])
        thread = Thread(target=thread_faceLandmarkDet)
        thread.start()
            
    
def crop_face_based_on_masked_image(masked_image):
    # Convert the image to grayscale
    gray_image = cvtColor(masked_image, COLOR_BGR2GRAY)

    # Find the coordinates of non-black (colored) pixels
    coords = findNonZero(gray_image)

    if coords is not None:
        # Get the bounding box of the non-black pixels
        x_min, y_min, w, h = boundingRect(coords)
        x_max = x_min + w
        y_max = y_min + h

        # Crop the masked image using these boundaries
        cropped_face = masked_image[y_min:y_max, x_min:x_max]

        return cropped_face
    else:
        print("No colored pixels found")
        return None
