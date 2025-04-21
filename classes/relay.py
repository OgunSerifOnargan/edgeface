from multiprocessing import Queue
from threading import Thread
from services.json_utils import read_json
from services.signalServices import telegram_send_message
from multiprocessing import Queue
from threading import Thread

class relay():
    def __init__(self, relay_id, IP):
        self.relay_id = relay_id
        self.IP = IP
        self.post_queue_relayDoor = Queue()
        self.relay_IP_updater_queue = Queue()
        # self.headers = {'Content-Type': 'application/json'}
        # self.session = Session()
        # self.session.headers.update(self.headers)

    # def send_signal_to_esp32(self):
    #     url = f"http://{self.IP}:600/signal"
    #     data = {'signal': "EventA"}
    #     try:
    #         response = self.session.post(url, json=data)
    #         print(f"relay: kapı tetiklenmesi tamamlandı: {datetime.now()}")
    #     except RequestException as e:
    #         print(f"Error: {e} relay")

    def post_result_to_relayDoor(self, client):
        config_system = read_json("facefinder/config/system.json")
        def post_result_to_relayDoor_in_thread():
            try:
                topic = f"relay_{self.relay_id}/trigger"
                payload = {'signal': "EventA"}
                client.publish(topic, payload=str(payload))  # MQTT mesajı gönder
            except Exception as e:
                text = f"!!!!!!!!!!!!!!!!relay opening door error!!!!!!!!!!!!!: {e}"
                if config_system["DEBUG_MODE"]:
                    telegram_send_message(text, None)

        thread = Thread(target=post_result_to_relayDoor_in_thread)
        thread.start()
