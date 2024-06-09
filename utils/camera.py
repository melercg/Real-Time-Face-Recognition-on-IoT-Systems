from cv2 import INTER_AREA
from enum import Enum
class CameraUtils():
    def __init__(self):
        self.Scale_X = 0.3
        self.Scale_Y = 0.3
        self.Capture_Src= 0
class Colors(Enum):
    primary  = (247, 84, 4)
    secondary = (249, 132, 4)
    third = (247, 253, 4)
    black = (0,0,0)
    white =(255,255,255)