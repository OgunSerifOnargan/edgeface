import json
import platform
import paho.mqtt.client as mqtt
import threading
import time
from services.json_utils import read_json
from services.signalServices import get_raspi_ip, telegram_send_message

class MQTTClient:
    def __init__(self, doors=None, broken_nodemcu_id_list=None, broken_relay_id_list=None, not_found_relay_list=None, not_found_station_list=None, mode="station"):
        self.config_system = read_json("facefinder/config/system.json")
        self.doors = doors
        self.mode = mode
        self.broker_IP = get_raspi_ip()
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        if self.mode == "main":
            self.LWT_TOPICS = self.get_lwt_topics()
        self.last_online_time = {}
        self.heartbeat_timeout = 120  # 120 saniye
        self.payload_door = str({'signal': "EventA"})
        self.broken_nodemcu_id_list = broken_nodemcu_id_list
        self.broken_relay_id_list = broken_relay_id_list
        self.not_found_relay_list = not_found_relay_list
        self.not_found_station_list = not_found_station_list
    
    def get_lwt_topics(self):
        topics = []
        for _, adoor in self.doors.items():
            if adoor.station_indoor_id:
                topics.append(adoor.station_indoor_lwt_topic)
                topics.append(adoor.station_indoor_ble_topic)
            if adoor.station_outdoor_id:
                topics.append(adoor.station_outdoor_lwt_topic)
                topics.append(adoor.station_outdoor_ble_topic)
        return topics
    
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
            if self.mode=="main":
                for topic in self.LWT_TOPICS:
                    client.subscribe(topic)
                    print(f"Subscribed to: {topic}")
        else:
            print(f"Connection failed with code {rc}")
    
    def on_message(self, client, userdata, msg):
        topic = msg.topic
        device_type, device_id = topic.split("/")[0].split("_")
        
        if msg.payload.decode() == "online":
            self.last_online_time[f"{device_type}_{int(device_id)}"] = time.time()
        
        if topic.endswith("/ble"):
            for adoor in self.doors.values():
                if str(adoor.station_indoor_id) == device_id and adoor.station_indoor:
                    self.publish(f"relay_{adoor.relay_id}/trigger", self.payload_door)
                    self.lcd_control("BLE_ENTRANCE", "BLE ENTRANCE", device_id)
                    telegram_send_message("BLE çıkışı yapıldı", None)
                if str(adoor.station_outdoor_id) == device_id and adoor.station_outdoor:
                    self.publish(f"relay_{adoor.relay_id}/trigger", self.payload_door)
                    self.lcd_control("BLE_ENTRANCE", "BLE ENTRANCE", device_id)
                    telegram_send_message("BLE girişi yapıldı", None)
    
    def on_disconnect(self, client, userdata, rc):
        print(f"Disconnected from MQTT Broker with code {rc}")
    
    def publish(self, topic, payload):
        self.client.publish(topic, payload)
    
    def lcd_control(self, message1, message2, station_id):
        if message1:
            topic = f"nodemcu_{station_id}/printLcd"
            data = json.dumps({"1:": message1, "2:": message2})
            self.client.publish(topic, data)
    
    def monitor_online_status(self):
        while True:
            current_time = time.time()
            for device, last_time in list(self.last_online_time.items()):
                if current_time - last_time > self.heartbeat_timeout:
                    device_type, device_id = device.split("_")
                    device_id = int(device_id)
                    if device_type == "nodemcu" and device_id not in self.broken_nodemcu_id_list:
                        self.broken_nodemcu_id_list.append(device_id)
                    elif device_type == "relay" and device_id not in self.broken_relay_id_list:
                        self.broken_relay_id_list.append(device_id)
                    del self.last_online_time[device]
            time.sleep(10)
    
    def start(self):
        self.client.connect(self.broker_IP, 1883, 60)
        mqtt_thread = threading.Thread(target=self.client.loop_forever, daemon=True)
        monitor_thread = threading.Thread(target=self.monitor_online_status, daemon=True)
        mqtt_thread.start()
        if self.mode == "main":
            monitor_thread.start()
        print("MQTT Client running in the background...")
