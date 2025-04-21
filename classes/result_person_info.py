from json import dumps, JSONEncoder
from requests import post, RequestException
from uuid import uuid4
from cv2 import imencode
from base64 import b64encode
from services.json_utils import read_json

class result_person_info:

    def __init__(self, camId, current_face, usage_reason="image"):
        self.omega_face_post = read_json("facefinder/config/omega_face_post.json")

        self.usage_reason = usage_reason
        if self.usage_reason == "image":
            self.camId = camId
            self.face_img = current_face.final_face_img
            self.face_img_base64 = _set_img_base64(self.face_img)
            self.name = current_face.final_printed_name
            self.trackerId = 0

            self.body_img = current_face.bestImg_dist_pair["defaultFrame"]
            self.body_img_base64 = _set_img_base64(self.body_img)
            self.uid_for_img_face = self.generate_uid()
            self.uid_for_img_body = self.generate_uid()
        if self.usage_reason == "rfid":
            self.camId = camId
            self.name = current_face.final_printed_name

    def construct_body_info(self):
        body = {
                "PostCrFormAnswers": [
                    {
                    "ID": -1, #// her zaman -1
                    "CRF_FIELDS_ID": self.omega_face_post["CRF_FIELD_TEXT_ID"],
                    "CRF_FORMS_ID": self.omega_face_post["CRF_FORMS_ID"],
                    "ROWDATARAW": self.name,  #//// formun ilk fieldı, buraya label bilgisini yazacağız
                    "ROWDATARAW2": "",
                    "rowState": 0
                    },
                    {
                    "ID": -1, #// her zaman -1
                    "CRF_FIELDS_ID": self.omega_face_post["CRF_FIELD_TEXT_ID"], #///Formun 2.fieldı buraya dosya göndereceğimin bilgisini yazacağım. 
                    "CRF_FORMS_ID": self.omega_face_post["CRF_FORMS_ID"],
                    "ROWDATARAW": "", #//her zaman boş
                    "ROWDATARAW2": self.uid_for_img_face, #// göndermeden önce ürettiğimiz UID
                    "rowState": 0
                    },
                    {
                    "ID": -1,
                    "CRF_FIELDS_ID": self.omega_face_post["CRF_FIELDS_ID_trackerId"],
                    "CRF_FORMS_ID": self.omega_face_post["CRF_FORMS_ID"],
                    "ROWDATARAW": str(self.trackerId), #// trackinID değeri neyse o, string alır
                    "ROWDATARAW2": "",
                    "rowState": 0
                    },
                    {
                    "ID": -1,
                    "CRF_FIELDS_ID": self.omega_face_post["CRF_FIELDS_ID_uuidImg"],
                    "CRF_FORMS_ID": self.omega_face_post["CRF_FORMS_ID"],
                    "ROWDATARAW": "",
                    "ROWDATARAW2": self.uid_for_img_body, #// göndermeden önce ürettiğimiz UID2 
                    "rowState": 0
                    },
                    {
                    "ID": -1,
                    "CRF_FIELDS_ID": self.omega_face_post["CRF_FIELDS_ID_inout"],
                    "CRF_FORMS_ID": self.omega_face_post["CRF_FORMS_ID"],
                    "ROWDATARAW": self.camId, #//in - out değeri, 0 out olsun 1 in olsun..
                    "ROWDATARAW2": "",
                    "rowState": 0
                    },
                    {
                    "ID": -1,
                    "CRF_FIELDS_ID": self.omega_face_post["CRF_FIELDS_ID_camId"],
                    "CRF_FORMS_ID": self.omega_face_post["CRF_FORMS_ID"],
                    "ROWDATARAW": self.camId, #//cam ID
                    "ROWDATARAW2": "",
                    "rowState": 0
                    }

                ],
                "PUBLICTOKEN": self.omega_face_post["PUBLIC_TOKEN"]
                }
        self.body_info = body

    def construct_body_img(self):
        body = {
                "UploadList": [
                    {
                    "BatchID": self.uid_for_img_face, #// bu dosya gönderimi için yukarda oluşturduğun file uid
                    "FileDetails": [
                        {
                        "FILENAME": f"{self.name}_face.png",
                        "RESOURCEID": self.omega_face_post["CRF_FIELD_TEXT_ID"], #// bu önemli, formun 2.fieldının ID’si
                        "IMGBASE64": self.face_img_base64,
                        "ARCHID": -1, #// her zaman -1
                        "ARCHIVECONTEXTID": -1, #// her zaman -1
                        
                        }
                    ]
                    },
                    {
                    "BatchID": self.uid_for_img_body, #// bu dosya gönderimi için yukarda oluşturduğun file uid
                    "FileDetails": [
                        {
                        "FILENAME": f"{self.name}_body.png",
                        "RESOURCEID": self.omega_face_post["CRF_FIELDS_ID_uuidImg"], #// bu önemli, formun 4.fieldının ID’si
                        "IMGBASE64": self.body_img_base64,
                        "ARCHID": -1, #// her zaman -1
                        "ARCHIVECONTEXTID": -1, #// her zaman -1
                        
                        }
                    ]
                    }
                ]
        }
        self.body_img = body

    def send_post_request(self, url, logger):
        final_url = self.omega_face_post["BASE_URL"] + url
        headers = {"Content-Type": "application/json"}
        try:
            if url == "PostCRFormAnswersPublic":
                self.body_info_json = dumps(self.body_info)
                response = post(final_url, data=self.body_info_json, headers=headers, verify=False)
            elif url == "PostCRFormAnswersPublicFileUpload":
                self.body_img_json = dumps(self.body_img)
                response = post(final_url, data=self.body_img_json, headers=headers, verify=False)
            response.raise_for_status()
        except RequestException as e:
            logger.error(f"Error sending request to {url}: {e}")
            raise

    def generate_uid(self):
        uid = str(uuid4())
        return uid

    def construct_body_for_post(self):
        self.construct_body_info()
        self.construct_body_img()

    def send_post_to_db(self, logger):
        self.send_post_request('PostCRFormAnswersPublic', logger)
        self.send_post_request('PostCRFormAnswersPublicFileUpload', logger)




    class CustomEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, result_person_info):
                return obj

            return JSONEncoder.default(self, obj)
            
def _set_img_base64(img):
    # Encode the image array to bytes
    _, buffer = imencode('.jpg', img)
    # Convert the bytes to base64 string
    base64_str = b64encode(buffer).decode('utf-8')
    return base64_str

