from time import time

class face():
    def __init__(self, trackerID, face_img, st, dlib_encodedVector, full_name, printed_name, dlib_result):
        self.trackerId = trackerID
        self.face_img = face_img
        self.bestImg_dist_pair = {"img": None,
                                  "min_val": 9999,
                                  "min_key": "",
                                  "defaultFrame":None}
        self.unknown_counter = 0  #10
        self.huge_model_trigger_counter = 0
        self.st = st
        self.et = None
        self.finalization_time = None
        self.finalizer_model = None

        self.temp_face_img = face_img
        self.temp_full_name = full_name
        self.temp_printed_name = printed_name
        self.temp_dlib_encodedVector = dlib_encodedVector
        self.temp_facenet_encodedVector = None
        self.temp_dlib_result = dlib_result
        self.temp_facenet_result = {"dlib_first_min_val":None,
                                  "dlib_first_min_key":None,
                                  "dlib_second_min_val":None,
                                  "dlib_second_min_key":None
                                  }

        self.final_face_img = None
        self.final_full_name = None
        self.final_printed_name = None
        self.final_dlib_encodedVector = None
        self.final_facenet_encodedVector = None
        self.final_dlib_result = {"dlib_first_min_val":None,
                                  "dlib_first_min_key":None,
                                  "dlib_second_min_val":None,
                                  "dlib_second_min_key":None
                                  }
        self.final_facenet_result = {"facenet_first_min_val":None,
                                  "facenet_first_min_key":None,
                                  "facenet_second_min_val":None,
                                  "facenet_second_min_key":None
                                  }

        self.ready_to_finalize = False
        
    def calculate_finalization_time(self):
        self.et = time()
        self.finalization_time = self.et-self.st
        print("Finalization time: ", self.finalization_time)

    def update_dlib_results(self, dlib_encodedVector, face_img, full_name, printed_name, dlib_result):
        self.temp_dlib_encodedVector = dlib_encodedVector
        self.temp_face_img = face_img
        self.temp_full_name = full_name
        self.temp_printed_name = printed_name
        self.temp_dlib_result = dlib_result


    def update_bestImg_dist_pair(self, defaultFrame):
        if self.temp_dlib_result["dlib_first_min_val"] < self.bestImg_dist_pair["min_val"]:
            self.bestImg_dist_pair["min_val"] = self.temp_dlib_result["dlib_first_min_val"]
            self.bestImg_dist_pair["img"] = self.temp_face_img
            self.bestImg_dist_pair["defaultFrame"] = defaultFrame
            self.bestImg_dist_pair["min_key"] = self.temp_dlib_result["dlib_first_min_key"]

    def finalize_face_info(self, model_name):
        self.final_face_img = self.temp_face_img
        self.final_full_name = self.bestImg_dist_pair["min_key"]
        self.final_printed_name = self.temp_printed_name
        self.final_dlib_encodedVector = self.temp_dlib_encodedVector
        self.final_dlib_result = self.temp_dlib_result
        if model_name == "facenet":
            self.final_facenet_encodedVector = self.temp_facenet_encodedVector
            self.final_facenet_result = self.temp_facenet_result

    def update_counters(self, unknown_counter, huge_model_trigger_counter):
        if unknown_counter is not None:
            if unknown_counter == 0:
                self.unknown_counter = 0
            else:
                self.unknown_counter += unknown_counter
        if huge_model_trigger_counter is not None:
            if huge_model_trigger_counter == 0:
                self.huge_model_trigger_counter = 0
            else:
                self.huge_model_trigger_counter += huge_model_trigger_counter