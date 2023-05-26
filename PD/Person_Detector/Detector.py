import cv2
import numpy as np
from Configurations import Configurations
from utils import * 


class personDetector(Configurations):

    def __init__(self,logs):
        super().__init__()
        self.logs = logs
        self.logs.UpdateLogs("Initializing Person Detector Module")
        self.cfg = self.cfg_path
        self.classes = self.loadClassNames(self.names_path)
        self.setColors(self.classes)
        self.weights_path = self.weights_path_person
        self.person_model = self.loadPersonModelWeights()
        self.logs.UpdateLogs("Person Model Initialized and Classes loaded")
        return None

    def setColors(self, classes):
        self.colors = np.random.uniform(0,255, size=(len(classes),3))
    
    def loadClassNames(self, names_path):
        self.logs.UpdateLogs("Loading Classnames")
        try:
            classes = []
            with open(names_path, "r") as f:
                classes = [line.strip() for line in f.readlines()]
            self.logs.UpdateLogs("Class Names Loaded")
            return classes
        except Exception as e:
            self.logs.UpdateErrorLogs('Failed to Load Class Names')
            self.logs.UpdateErrorLogs(str(e))
            exit(0)
        

    def loadPersonModelWeights(self):
        
        try:
            self.logs.UpdateLogs("Loading Model Weight and Config")
            # configure the network model
            net = cv2.dnn.readNetFromDarknet(self.cfg, self.weights_path)
            # Configure the network backend
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            self.logs.UpdateLogs("Loaded Model Weight and Config")
            return net
        except Exception as e:
            self.logs.UpdateErrorLogs('Model Loading Failed')
            self.logs.UpdateErrorLogs(str(e))
            exit(0)  

    def loadImage(self,sourceImg):
         try:
            self.logs.UpdateLogs("Reading Source Image")
            img = cv2.imread(sourceImg)
            height,width, channels = img.shape
            return img, height,width,channels
         except Exception as e:
             self.logs.UpdateErrorLogs("Failed to Read Image")
             self.logs.UpdateErrorLogs(str(e))
             exit(0)
                
    def processImage(self,sourceImg):
            try:
                img, height, width, channels = self.loadImage(sourceImg)
                blob,detected_persons = self.detectObject(img)
                imgCopy = img.copy()
                boxes, classIds, confidence_scores = self.getBBoxDetails(height, width, detected_persons,imgCopy)
                if len(boxes) > 0:
                    self.applyNMS(img, boxes, classIds, confidence_scores)  
                return
                 
            except Exception as e:
                self.logs.UpdateErrorLogs('Model failing to detect')
                self.logs.UpdateErrorLogs("Error Msg:  " +str(e))
                exit(0)

    def applyNMS(self, img, boxes, classIds, confidence_scores):
        self.logs.UpdateLogs("Applying NMS")
        indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, self.confThreshold, self.nmsThreshold) # Apply Non-Max Suppression
        print(indices)
        print(boxes)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if len(indices) > 0:
                for i in indices.flatten():
                    if i in indices:
                        x,y,w,h = boxes[i]
                        label = str(self.classes[classIds[i]])
                        color = self.colors[i]
                        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                        cv2.putText(img, label,(x, y - 5), font, 1, color, 1)
        SaveImg(os.path.join(self.images_path_person,"WithNMS.jpg"),img)

    def getBBoxDetails(self, height, width, detected_persons,img):
        self.logs.UpdateLogs("Processing BBOX")
        boxes = []
        classIds = []
        confidence_scores = []
        for output in detected_persons:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId] 
                if classId in self.person_class_index:
                    if confidence > self.confThreshold:
                        w,h = int(det[2]*width) , int(det[3]*height)
                        x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                        color = self.colors[1]
                        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                        cv2.putText(img, "Person",(x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
                        boxes.append([x,y,w,h])
                        classIds.append(classId)
                        confidence_scores.append(float(confidence))
        SaveImg(os.path.join(self.images_path_person,"WithoutNMS.jpg"),img)
        return boxes,classIds,confidence_scores

    def detectObject(self, img):
        self.logs.UpdateLogs("Detect Objects")
        blob = cv2.dnn.blobFromImage(img, self.scale_factor , (self.inputsize, self.inputsize), [0, 0, 0], self.swapRB, crop=False)
        self.person_model.setInput(blob)
        layersNames = self.person_model.getLayerNames()
        outputNames = [(layersNames[i- 1]) for i in self.person_model.getUnconnectedOutLayers()]
        detected_persons = self.person_model.forward(outputNames)
        return blob,detected_persons


