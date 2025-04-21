from time import time, sleep
from classes.relay import relay
from classes.station import station
from classes.door import create_station_processes, door
from services.json_utils import read_json
from services.signalServices import ping_device, telegram_send_message
import netifaces
config_system = read_json("facefinder/config/system.json")

def initialize_doors(doors, registered_doors):
    for door_id, registered_door in registered_doors.items():
        doors[door_id] = door(door_id, registered_door)
    return doors

def get_ip_range():
    try:
        private_ip_ranges = ("192.168.", "10.", "172.16.", "172.17.", "172.18.", "172.19.", "172.20.", 
                             "172.21.", "172.22.", "172.23.", "172.24.", "172.25.", "172.26.", "172.27.", 
                             "172.28.", "172.29.", "172.30.", "172.31.")
        
        interfaces = netifaces.interfaces()

        for interface in interfaces:
            addrs = netifaces.ifaddresses(interface).get(netifaces.AF_INET)
            if addrs:
                for addr in addrs:
                    ip = addr['addr']
                    if ip.startswith(private_ip_ranges):
                        base_ip = ".".join(ip.split(".")[:3]) + ".0/24"
                        return base_ip
        print("No valid LAN IP found!")  # Debug
    except Exception as e:
        print(f"Error: {e}")
    return "192.168.0.0/24"

def find_device(found_devices, device_id, device_type):
    for found_device in found_devices.values():
        if found_device["type"] == device_type and found_device["id"] == str(device_id):
            return found_device
        else:
            pass

def find_related_relay_with_station(related_nodemcu, doors):
    for door_id, adoor in doors.items():
        if str(adoor.station_indoor_id) == related_nodemcu["id"] or str(adoor.station_outdoor_id) == related_nodemcu["id"]:
            return adoor.relay

def add_found_station_to_doors(doors, related_cam, related_nodemcu, not_found_station_list, landmark_queue_dict, face_recognizer_queue_dict, faceDet_to_landmark_queue, faceID_to_faceRecognizer_queue, json_lock):
    for door_id, adoor in doors.items():
        if adoor.relay:
            if related_nodemcu["id"] == str(adoor.station_indoor_id):
                adoor.station_indoor = station(adoor.station_indoor_id, related_cam["ip"], related_nodemcu["ip"], adoor.relay.relay_id, adoor.relay.IP, landmark_queue_dict, face_recognizer_queue_dict, faceDet_to_landmark_queue, faceID_to_faceRecognizer_queue, json_lock)
                not_found_station_list.remove(adoor.station_indoor_id)
            if related_nodemcu["id"] == str(adoor.station_outdoor_id):
                adoor.station_outdoor = station(adoor.station_outdoor_id, related_cam["ip"], related_nodemcu["ip"], adoor.relay.relay_id, adoor.relay.IP, landmark_queue_dict, face_recognizer_queue_dict, faceDet_to_landmark_queue, faceID_to_faceRecognizer_queue, json_lock)
                not_found_station_list.remove(adoor.station_outdoor_id)
    return doors, not_found_station_list

def add_found_relay_to_doors(doors, related_relay):
    for door_id, adoor in doors.items():
        if related_relay["id"] == str(adoor.relay_id):
            adoor.relay = relay(related_relay["id"], related_relay["ip"])
    return doors


def add_found_devices_to_doors(doors, found_devices, not_found_station_list, not_found_relay_list, landmark_queue_dict, face_recognizer_queue_dict, faceDet_to_landmark_queue, faceID_to_faceRecognizer_queue, json_lock):
    for door_id, adoor in doors.items():
        if adoor.relay_id is not None:
            related_relay = find_device(found_devices, adoor.relay_id, "relay")
            if related_relay is not None:
                adoor.relay = relay(related_relay["id"], related_relay["ip"])
                if adoor.station_indoor_id is not None:
                    related_nodemcu_indoor = find_device(found_devices, adoor.station_indoor_id, "nodemcu")
                    related_cam_indoor = find_device(found_devices, adoor.station_indoor_id, "cam")
                    if related_nodemcu_indoor is not None and related_cam_indoor is not None:
                        adoor.station_indoor = station(adoor.station_indoor_id, related_cam_indoor["ip"], related_nodemcu_indoor["ip"], adoor.relay.relay_id, adoor.relay.IP, landmark_queue_dict, face_recognizer_queue_dict, faceDet_to_landmark_queue, faceID_to_faceRecognizer_queue, json_lock)
                        if adoor.station_indoor_id in not_found_station_list:
                            not_found_station_list.remove(adoor.station_indoor_id)
                    else:
                        print(f"Station_indoor is not found for door_id: {door_id}, station_id: {adoor.station_indoor_id}")
                        if adoor.station_indoor_id not in not_found_station_list:
                            not_found_station_list.append(adoor.station_indoor_id)
                if adoor.station_outdoor_id is not None:
                    related_nodemcu_outdoor = find_device(found_devices, adoor.station_outdoor_id, "nodemcu")
                    related_cam_outdoor = find_device(found_devices, adoor.station_outdoor_id, "cam")
                    if related_nodemcu_outdoor is not None and related_cam_outdoor is not None:
                        adoor.station_outdoor = station(adoor.station_outdoor_id, related_cam_outdoor["ip"], related_nodemcu_outdoor["ip"], adoor.relay.relay_id, adoor.relay.IP, landmark_queue_dict, face_recognizer_queue_dict, faceDet_to_landmark_queue, faceID_to_faceRecognizer_queue, json_lock)
                        if adoor.station_outdoor_id in not_found_station_list:
                            not_found_station_list.remove(adoor.station_outdoor_id)
                    else:
                        print(f"Station_indoor is not found for door_id: {door_id}, station_id: {adoor.station_outdoor_id}")
                        if adoor.station_outdoor_id not in not_found_station_list:
                            not_found_station_list.append(adoor.station_outdoor_id)
            else:
                print(f"Relay is not found for door_id: {door_id}")
                if adoor.relay_id not in not_found_relay_list:
                    not_found_relay_list.append(adoor.relay_id)
                    if adoor.station_outdoor_id:
                        not_found_station_list.append(adoor.station_outdoor_id)
                    if adoor.station_indoor_id:
                        not_found_station_list.append(adoor.station_indoor_id)
        else:
            print(f"relay is not attained for door_id: {door_id}")
            if adoor.relay_id not in not_found_relay_list:
                not_found_relay_list.append(adoor.relay_id)
    return doors, not_found_station_list, not_found_relay_list

def init_door_threads_and_processes(doors, threads, processes, stop_event, broken_cam_list):
    for adoor in doors.values():
        if adoor.relay:
            if adoor.station_indoor:
                processes = create_station_processes(stop_event, processes, adoor, adoor.station_indoor_id, broken_cam_list, "indoor")
            if adoor.station_outdoor:
                processes = create_station_processes(stop_event, processes, adoor, adoor.station_outdoor_id, broken_cam_list, "outdoor")
    return threads, processes

def check_devices(doors, lost_counter, broken_nodeMCU_list, broken_relay_list):
    for door_id, adoor in doors.items():
        if adoor.station_indoor is not None:
            if ping_device(adoor.station_indoor.nodeMCU.nodeMCU_IP):
                text = f"Station is reachable: {adoor.station_indoor_id}"
                lost_counter["station_indoor"] = 0
            else:
                print("connection lost ", time()," ",lost_counter["station_indoor"], "nodemcu")
                lost_counter["station_indoor"] += 1
                if lost_counter["station_indoor"] >= 3:
                    text = f"Station is NOT reachable: {adoor.station_indoor_id}"
                    if config_system["DEBUG_MODE"]:
                        telegram_send_message(text, None)
                    if adoor.station_indoor_id not in broken_nodeMCU_list:
                        broken_nodeMCU_list.append(adoor.station_indoor_id)

        if adoor.station_outdoor is not None:
            if ping_device(adoor.station_outdoor.nodeMCU.nodeMCU_IP):
                text = f"Station is reachable: {adoor.station_outdoor_id}"
                lost_counter["station_outdoor"] = 0
            else:
                print("connection lost ", time()," ",lost_counter["station_outdoor"], "nodemcu")
                lost_counter["station_outdoor"] += 1
                if lost_counter["station_outdoor"] >= 3:
                    text = f"Station is NOT reachable: {adoor.station_outdoor_id}"
                    if config_system["DEBUG_MODE"]:
                        telegram_send_message(text, None)
                    if adoor.station_outdoor_id not in broken_nodeMCU_list:
                        broken_nodeMCU_list.append(adoor.station_outdoor_id)

        if adoor.relay is not None:
            if ping_device(adoor.relay.IP):
                text = f"Relay is reachable: {adoor.relay_id}"
                lost_counter["relay"] = 0
            else:
                print("connection lost ", time()," ",lost_counter["relay"], " relay")
                lost_counter["relay"] += 1
                if lost_counter["relay"] >= 3:
                    text = f'Relay is NOT reachable: {adoor.relay_id}'
                    if config_system["DEBUG_MODE"]:
                        telegram_send_message(text, None)
                    if adoor.relay_id not in broken_relay_list:
                        broken_relay_list.append(adoor.relay_id) 

def repair_broken_devices(updated_devices_on_lan, broken_cam_list, broken_nodeMCU_list, broken_relay_list, doors):
    for adoor in doors.values():
        if adoor.station_indoor_id in broken_cam_list:
            related_cam_indoor = find_device(updated_devices_on_lan, adoor.station_indoor_id, "cam")
            if related_cam_indoor:
                adoor.station_indoor.camera.cam_IP = related_cam_indoor["ip"]
                adoor.station_indoor.camera.camIP_updater_queue.put(related_cam_indoor["ip"])
                broken_cam_list.remove(int(related_cam_indoor["id"]))

        if adoor.station_indoor_id in broken_nodeMCU_list:
            related_nodemcu_indoor = find_device(updated_devices_on_lan, adoor.station_indoor_id, "nodemcu")
            if related_nodemcu_indoor:
                adoor.station_indoor.nodeMCU.nodeMCU_IP = related_nodemcu_indoor["ip"]
                adoor.station_indoor.nodeMCU.nodeMCU_IP_updater_queue.put(related_nodemcu_indoor["ip"])
                broken_nodeMCU_list.remove(int(related_nodemcu_indoor["id"]))
                print(f"Bir nodemcu tamir edildi. : {adoor.station_indoor_id}")

        if adoor.station_outdoor_id in broken_cam_list:
            related_cam_outdoor = find_device(updated_devices_on_lan, adoor.station_outdoor_id, "cam")
            if related_cam_outdoor:
                adoor.station_outdoor.camera.cam_IP = related_cam_outdoor["ip"]
                adoor.station_outdoor.camera.camIP_updater_queue.put(related_cam_outdoor["ip"])
                broken_cam_list.remove(int(related_cam_outdoor["id"]))

        if adoor.station_outdoor_id in broken_nodeMCU_list:
            related_nodemcu_outdoor = find_device(updated_devices_on_lan, adoor.station_outdoor_id, "nodemcu")
            if related_nodemcu_outdoor:
                adoor.station_outdoor.nodeMCU.nodeMCU_IP = related_nodemcu_outdoor["ip"]
                adoor.station_outdoor.nodeMCU.nodeMCU_IP_updater_queue.put(related_nodemcu_outdoor["ip"])
                broken_nodeMCU_list.remove(int(related_nodemcu_outdoor["id"]))
                print(f"Bir nodemcu tamir edildi. : {adoor.station_outdoor_id}")

        if adoor.relay_id in broken_relay_list:
            related_relay = find_device(updated_devices_on_lan, adoor.relay_id, "relay")
            if related_relay:
                adoor.relay.IP = related_relay["ip"]
                adoor.station_indoor.relay.IP = related_nodemcu_outdoor["ip"]
                adoor.station_outdoor.relay.IP = related_nodemcu_outdoor["ip"]
                if adoor.station_indoor:
                    adoor.station_indoor.relay_IP_updater_queue.put(related_relay["ip"])
                if adoor.station_outdoor:
                    adoor.station_outdoor.relay_IP_updater_queue.put(related_relay["ip"])
                
                print(broken_relay_list)
                broken_relay_list.remove(int(related_relay["id"]))
                print(f'Bir relay tamir edildi. : {related_relay["id"]}')
            
    return doors, broken_cam_list, broken_nodeMCU_list, broken_relay_list

def start_new_processes_and_threads(processes, threads):
    for process_name, process in processes.items():
        if not process.is_alive():
            process.start()
            if config_system["DEBUG_MODE"]:
                print(process.name, " has been started")
            sleep(0.1)
    for thread_name, thread in threads.items():
        if not thread.is_alive():
            thread.daemon = True
            thread.start()
            sleep(0.1)
    return processes, threads
        
def print_broken_and_not_found_devices(broken_cam_list, broken_nodeMCU_list, broken_relay_list, not_found_station_list, not_found_relay_list):
    print(f"borken_cam_list: {broken_cam_list}")
    print(f"broken_nodeMCU_list: {broken_nodeMCU_list}" )
    print(f"broken_relay_list: {broken_relay_list}")
    print(f"not_found_station_list: {not_found_station_list}")
    print(f"not_found_relay_list: {not_found_relay_list}")

def send_telegram_message_for_broken_devices(broken_cam_list, broken_nodeMCU_list, broken_relay_list):
    if broken_cam_list:
        text = f"Cams are NOT reachable: {broken_cam_list}"
        if config_system["DEBUG_MODE"]:
            telegram_send_message(text, None)
    if broken_nodeMCU_list:
        text = f"nodemcus are NOT reachable: {broken_nodeMCU_list}"
        if config_system["DEBUG_MODE"]:
            telegram_send_message(text, None)  
    if broken_relay_list:
        text = f"relays are NOT reachable: {broken_relay_list}"
        if config_system["DEBUG_MODE"]:
            telegram_send_message(text, None)    

def find_and_connect_not_found_relay_to_doors(doors, not_found_relay_list, updated_devices_on_lan):
    for not_found_relay in not_found_relay_list:
        related_relay = find_device(updated_devices_on_lan, not_found_relay, "relay")
        if related_relay:
            doors = add_found_relay_to_doors(doors, related_relay)
            not_found_relay_list.remove(not_found_relay)
    return doors, not_found_relay_list




# def find_new_station_not_found_station_list(scan_IP_interval, not_found_station_list):
#     if not_found_station_list:
#         devices = scan_network(scan_IP_interval)
#         new_stations, new_relays = create_stations(devices)
#         new_stations_id_list = list(new_stations.keys())
#         newly_found_station_id_list = (set(new_stations_id_list) - set(current_station_id_list))

#         new_relays_id_list = list(new_relays.keys())
#         newly_found_relay_id_list = (set(new_relays_id_list) - set(current_relay_id_list))
#         return newly_found_station_id_list, new_stations, newly_found_relay_id_list, new_relays

# def create_stations(devices):
#     # Create a dictionary to hold stations with box-id as key
#     temp_stations = {}
#     stations = {}
#     relays = {}
#     current_relay_id = None
#     for device in devices.values():
#         if device["type"] == "relay":
#             current_relay_id = int(device["relay_id"])
#             relays.setdefault(current_relay_id, {})
#             relay_ip = device["ip"]
#             relays[current_relay_id] = relay(current_relay_id, device["ip"])
#             break
#     for device in devices.values(): 
#         if device["type"] == "relay":
#             continue  
#         else: 
#             current_station_id = int(device["station_id"])
#             temp_stations.setdefault(current_station_id, {})
#             if device["type"] == "nodemcu":
#                 temp_stations[current_station_id]["nodeMCU_ip"] = device["ip"]
#             if device["type"] == "cam":
#                 temp_stations[current_station_id]["cam_ip"] = device["ip"]

#     if temp_stations:
#         for station_id, temp_station in temp_stations.items():
#             if temp_station.get("cam_ip") and temp_station.get("nodeMCU_ip") and current_relay_id:
#                 stations[station_id] = station(station_id, temp_station["cam_ip"], temp_station["nodeMCU_ip"], current_relay_id, relay_ip)
#             else:
#                 if not temp_station.get("cam_ip"):
#                     print(f"Camera is not found for station {station_id}")
#                     continue
#                 if not temp_station.get("nodeMCU_ip"):
#                     print(f"NodeMCU is not found for station {station_id}")
#                     continue
#     else:
#         print("No device is found.Retrying after 60 sec...")
#         sleep(60)
#     return stations, relays 


