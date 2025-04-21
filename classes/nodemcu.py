from multiprocessing import Queue
from time import time
import netifaces

def get_local_ip():
    interfaces = netifaces.interfaces()
    for interface in interfaces:
        if interface != 'lo':  # Ignore loopback interface
            addresses = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addresses:
                return addresses[netifaces.AF_INET][0]['addr']

class nodemcu:
    def __init__(self, station_id, nodeMCU_IP):
        self.station_id = station_id
        self.nodeMCU_IP = nodeMCU_IP
        self.post_queue_nodeMCU = Queue()
        self.nodeMCU_IP_updater_queue = Queue()
        self.nodeMCU_post_fail_counter = 0
        self.prev_message = {"time": time(), "message": ""}
        self.ready_to_send = False

        self.message_array = ["", "", ""]
        self.headers = {'Content-Type': 'application/json'}



# Example usage:
# node = nodemcu(station_id=1, nodeMCU_IP="192.168.1.100")
# node.send_json_to_nodeMCU(known_people)
# node.lcd_control("Hos geldiniz", "Welcome!")

    # def send_json_to_nodeMCU(self, known_people):
    #     url = f"http://{self.nodeMCU_IP}:600/sync"
    #     json_data = dumps({"known_people": known_people}) 
    #     print(json_data)
    #     try:
    #         # Send request using session to reuse the connection
    #         response = self.session.post(url, data=json_data)
    #         print("Status Code:", response.status_code)
    #         print("Response Text:", response.text)
    #     except RequestException as e:
    #         print("Json post Connection error:", e)