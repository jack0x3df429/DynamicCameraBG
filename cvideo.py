import cv2
class   cvideo():
    def __init__(self,file,size=(1600,900)):
        self.capvideo=cv2.VideoCapture(file)
        self.size=size
    def read(self): 
        ret, frame = self.capvideo.read()        
        if not ret:
            self.capvideo.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return self.read()
        #else:
        #    frame=cv2.resize(frame, self.size, interpolation=cv2.BORDER_DEFAULT)
        return frame


if __name__ =="__main__":
    pass
    '''while True:
        c=cvideo("rick.mp4")

        cv2.imshow("my webcam",         c.read())
        cv2.waitKey()'''