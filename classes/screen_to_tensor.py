#
#
# File not used at this point
#

import mss
import numpy as np
import time

class MSSShot:
    
    # an object that continously yields screenshots as numpy arrays
    
    def __init__(self, top=False, left=False, width=False, height=False):
        self.top = top
        self.left = left
        self.width = width
        self.height = height
        self.quit = False
        self.init_time = time.time()
    
    def __call__(self):
        # if any params in self are missing, it will just shot the whole screen
        # yields (time passed since function was called, image as numpy array)
        run = True
        with mss.mss() as sct:
            if all([bool(i) for i in [self.width, self.height]]):
                monitor = {"top": self.top, "left": self.left, "width": self.width, "height": self.height}
            else:
                monitor = False
            if monitor:
                run = True
                while run:
                    image = np.array(sct.grab(monitor))
                    yield (time.time()-self.init_time, image)
                    # mss yields tensors of dim BGRA (Blue, Green, Red, Alpha)
                    if self.quit:
                        run = False
            else:
                run = True
                while run:
                    image = np.array(sct.grab(sct.monitors[1]))
                    yield (time.time()-self.init_time, image)
                    if self.quit:
                        run = False
        self.quit = False
    
    def quit_progr(self):
        self.quit = True
    
class CV2Shot:
    
    def __init__(self, f_rate):
        self.f_rate = f_rate
        self.init_time = time.time()
    
    def __call__(self, path):
        pass
        

if __name__ == '__main__':
    pass