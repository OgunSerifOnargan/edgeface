from multiprocessing import Queue, Process

class door:
    def __init__(self, door_id, registered_door):
        self.door_id = door_id
        self.station_indoor_id = registered_door["station_indoor_id"]
        self.station_outdoor_id = registered_door["station_outdoor_id"]
        self.relay_id = registered_door["relay_id"]
        self.station_indoor = None
        self.station_outdoor = None
        self.relay = None
        self.state_fully_connected = False
        self.web_server_rfid_queue = Queue()
        if self.station_indoor_id:
            self.station_indoor_lwt_topic = f"nodemcu_{self.station_indoor_id}/status"
            self.station_indoor_ble_topic = f"nodemcu_{self.station_indoor_id}/ble"
        if self.station_outdoor_id:
            self.station_outdoor_lwt_topic = f"nodemcu_{self.station_outdoor_id}/status"
            self.station_outdoor_ble_topic = f"nodemcu_{self.station_outdoor_id}/ble"
        self.relay_lwt_topic = f"relay_{self.relay_id}/status"

def find_name_by_card_uid(card_uid, known_card_uids, known_face_names):
    try:
        index = known_card_uids.index(card_uid)
        return known_face_names[index]
    except ValueError:
        return "Unknown"
    
def create_station_processes(stop_event, processes, adoor, station_id, broken_cam_list, indoor_outdoor):
    #TODO: 1: landmark_queue_dict station prosesine eklenecek
    x=1
    if indoor_outdoor == "indoor":
        if f"station_main_loop_{station_id}" not in processes.keys():
            processes[f"station_main_loop_{station_id}"] = Process(target=adoor.station_indoor.station_main_loop, 
                                                                    args=(stop_event, broken_cam_list), name=f"p_station_main_loop_{station_id}")
    else:
        if f"station_main_loop_{station_id}" not in processes.keys():
            processes[f"station_main_loop_{station_id}"] = Process(target=adoor.station_outdoor.station_main_loop, 
                                                                    args=(stop_event, broken_cam_list), name=f"p_station_main_loop_{station_id}")
    return processes
