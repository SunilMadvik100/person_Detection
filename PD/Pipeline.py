
from Configurations import Configurations
from utils import *
from utils import Logs
from utils import CalcTime
from Person_Detector import Detector as AI

class person_inference(Configurations):

    def __init__(self):
        super().__init__()
        self.logs = Logs()
        self.timer = CalcTime()
        self.personDetection = AI.personDetector(self.logs)

    def __call__(self):
        CreateFolder(self.images_path_person)
        self.timer.start()
        self.personDetection.processImage(self.sourcePath)
        self.timer.end()
        self.logs.UpdateLogs("Time took to process Image :" + str(self.timer.calculate()) + "s" )

if __name__ == "__main__":
    inference = person_inference()
    inference()
    