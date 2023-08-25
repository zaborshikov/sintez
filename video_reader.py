import cv2


class Video():
    def __init__(self, filename):
        cap = cv2.VideoCapture(filename)
        self.filename = filename
        self.readed = False
        self.length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.time = self.length/self.fps
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        if not cap.isOpened():
            print("The file doesn't exist, or it's not a video")
            raise FileExistsError
        else:
            self.cap = cap
    
    
    def read_frame(self, n=-1):
        current_frame = self.current_frame_number()
        if n != -1:
            self.set_frame(n)
        
        cap = self.cap
        self.readed = True

        if cap.isOpened():
            ret, frame = cap.read()
            if n != -1:
                self.set_frame(current_frame + 1)
            return [True, frame]
        return [False, None]
        
    
    
    def set_frame(self, n):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    
    
    def current_frame_number(self):
        return self.cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    
    def return_list(self, n=-1):
        current_frame = self.current_frame_number()
        startframe = 0
        
        if n != -1:
            startframe = n
        
        self.set_frame(startframe)
        frames = []
        frame = self.read_frame()
        
        while frame[0]:
            if self.current_frame_number() == self.length:
                break
            frames.append(frame)
            frame = self.read_frame()
        
        self.set_frame(current_frame)
        
        return frames