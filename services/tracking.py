# from cv2 import rectangle, TrackerCSRT_create
from cv2 import rectangle
# def calculate_average_box(boxes):
#     if not boxes:
#         return []
    
#     avg_box = [0, 0, 0, 0]
#     for box in boxes:
#         avg_box[0] += box[0]
#         avg_box[1] += box[1]
#         avg_box[2] += box[2]
#         avg_box[3] += box[3]
    
#     avg_box = [coord / len(boxes) for coord in avg_box]
#     return avg_box

def draw_tracking_bbox_on_frame(frame, bbox, bbox_type="xyxy", moving_average=False):
    if bbox_type == "xywh":
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    if bbox_type == "xyxy":
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[2]), int(bbox[3]))
    if moving_average:
        rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        rectangle(frame, p1, p2, (0, 255, 0), 2, 1)

# def initialize_tracker(frame, bbox_biggest_xywh):
#     tracker = TrackerCSRT_create()
#     ok = tracker.init(frame, bbox_biggest_xywh)
#     return tracker, ok