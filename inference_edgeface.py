import os
from classes.landmark_detector import landmark_detector
import cv2
import torch
from torchvision import transforms
from backbones import get_model
from torch.nn.functional import cosine_similarity

from processes.image_processing import preprocess_face_img

# --- SETTINGS ---
REFERENCE_IMAGE_DIRECTORY = '/Users/onarganogun/Downloads/repos/pdks-box/facefinder/known_face_folders/station_1'  # klasör yapısı: known_faces/Alice/1.jpg ...
MODEL_NAME = "edgeface_s_gamma_05"
CHECKPOINT_PATH = f'/Users/onarganogun/AdaFace/edgeface/checkpoints/edgeface_s_gamma_05.pt'
WEBCAM_INDEX = 1
SIMILARITY_THRESHOLD = 0.40  # cosine distance'e göre eşik

landmarkDetector = landmark_detector(usage_reason="realtime",
                                     landmark_queue_dict=None,
                                     station_id="2",
                                     up_threshold=0.5,
                                     down_threshold=2,
                                     left_threshold=3,
                                     right_threshold=0.45,
                                     display_mode=False)

landmarkDetector_for_ref = landmark_detector(usage_reason="image",
                                             landmark_queue_dict=None,
                                             station_id="2",
                                             up_threshold=0.5,
                                             down_threshold=2,
                                             left_threshold=3,
                                             right_threshold=0.45,
                                             display_mode=False)

# --- LOAD MODEL ---
model = get_model(MODEL_NAME)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# --- Cosine Distance ---
def cosine_distance(tensor1, tensor2):
    return 1 - cosine_similarity(tensor1, tensor2).item()

# --- Load & Encode Reference Faces ---
def encode_reference_faces(reference_dir):
    encoding_dict = {}
    for person_name in os.listdir(reference_dir):
        person_path = os.path.join(reference_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        embeddings = []
        for file in os.listdir(person_path):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            full_path = os.path.join(person_path, file)
            frame = cv2.imread(full_path)
            image, face_img, _, flag_face_skip, *_ = landmarkDetector_for_ref.crop_face_from_landmarks(frame)
            if face_img is None or len(face_img.shape) != 3:
                print("Warning: Skipping invalid face image.")
                continue
            face_img = preprocess_face_img(face_img)
            transformed = transform(face_img).unsqueeze(0)
            with torch.no_grad():
                embedding = model(transformed).detach().cpu()
                embeddings.append(embedding)

        if embeddings:
            mean_embedding = torch.mean(torch.stack(embeddings), dim=0)
            encoding_dict[person_name] = mean_embedding
            print(f"[ENCODED] {person_name} with {len(embeddings)} images.")

    return encoding_dict

# --- MAIN LOOP ---
def main():
    reference_encodings = encode_reference_faces(REFERENCE_IMAGE_DIRECTORY)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("Error: Webcam not found.")
        return

    print("Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Align the face from frame
        image, face_img, _, flag_face_skip, *_ = landmarkDetector.crop_face_from_landmarks(frame)
        if face_img is None or len(face_img.shape) != 3:
            print("Warning: Invalid face image.")
            continue
        face_img = preprocess_face_img(face_img)
        cv2.imshow("asda", face_img)
        transformed = transform(face_img).unsqueeze(0)
        with torch.no_grad():
            embedding = model(transformed).detach().cpu()

        best_match = None
        lowest_distance = float('inf')

        for name, ref_embedding in reference_encodings.items():
            dist = cosine_distance(embedding, ref_embedding)
            if dist < lowest_distance:
                lowest_distance = dist
                best_match = name

        if lowest_distance < SIMILARITY_THRESHOLD:
            cv2.putText(frame, f"{best_match} ({lowest_distance:.2f})", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Unknown ({lowest_distance:.2f})", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
