import json

class Configurations(object):

    json_path = "Configurations.json"


    def __init__(self):
        self.configs = Configurations.LoadJson()
        self.InferenceMode()
        self.getSourcePath()
        self.Load_PersonDetector_Configurations()
        return None


    @classmethod
    def LoadJson(cls):
        with open(Configurations.json_path, 'r') as fh:
            configs = json.load(fh)
        return configs

    def InferenceMode(self):
        self.inference_mode = self.configs["inference_mode"]
        return None
       
    def getSourcePath(self):
        self.sourcePath = self.configs["sourcePath"]
        return None

    def Load_PersonDetector_Configurations(self):
        self.cfg_path = self.configs["Person_Detection"]["Yolo_V3"]["cfg_path"] 
        self.names_path = self.configs["Person_Detection"]["Yolo_V3"]["names_path"]
        self.weights_path_person = self.configs["Person_Detection"]["Yolo_V3"]["weights_path"]
        self.images_path_person = self.configs["Person_Detection"]["Yolo_V3"]["images_path"]
        self.bbox_padding = self.configs["Person_Detection"]["Yolo_V3"]["bbox_padding"]
        self.confThreshold = self.configs["Person_Detection"]["Yolo_V3"]["Threshold_values"]["confThreshold"]
        self.nmsThreshold = self.configs["Person_Detection"]["Yolo_V3"]["Threshold_values"]["nmsThreshold"]
        self.inputsize = self.configs["Person_Detection"]["Yolo_V3"]["Threshold_values"]["input_size"]
        self.person_class_index = self.configs["Person_Detection"]["Yolo_V3"]["Threshold_values"]["person_class_index"]
        self.scale_factor = self.configs["Person_Detection"]["Yolo_V3"]["Threshold_values"]["scale_factor"]
        self.swapRB =self.configs["Person_Detection"]["Yolo_V3"]["Threshold_values"]["swapRB"]
        self.crop = self.configs["Person_Detection"]["Yolo_V3"]["Threshold_values"]["crop"]
        return None
    