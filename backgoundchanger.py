import cv2
import mediapipe as mp
import numpy as np
from cvideo import cvideo
import threading,time
def show(a):
    if np.max(a)==1: a=a*255
    a = cv2.resize(a, (a.shape[1]//4,a.shape[0]//4), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("a",a.astype(np.uint8))
    cv2.waitKey(0)
def info(a):
    print("%s\ndtype:%s\nmax:%d\nmin%d"%(a.shape,a.dtype,np.max(a),np.min(a)))
class backgoundchanger(threading.Thread):
    def __init__(self,file,Output_res):
        threading.Thread.__init__(self)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.BG_COLOR = (0, 255, 0) 
        self.MASK_COLOR = (255, 255, 255)
        self.file = file
        self.cap=cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.Output_res=Output_res
        self.Camera_res=(0,0)
        self.capvideo=cvideo(file,(self.Output_res[1],self.Output_res[0]))
        self.image=np.zeros((self.Output_res[0],self.Output_res[1],3))
        self.stop_flag=False
        self.frame_pos=[]
        #獲得相機圖
        #獲得分割遮罩
        #切下影像
        #獲得背景
        #合成
    def run(self):
        while not self.stop_flag:
            with self.mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
                #人像
                ret,image = self.cap.read()
                self.Camera_res = image.shape[:2]

                #拉至螢幕大小
                rate = self.Output_res[0]/self.Camera_res[0]
                image = cv2.resize(image, (int(self.Camera_res[1]*rate),int(self.Camera_res[0]*rate)), interpolation=cv2.INTER_CUBIC)
                self.Camera_res = image.shape[:2]
                image = image[
                (self.Camera_res[0]-self.Output_res[0])//2:(self.Camera_res[0]+self.Output_res[0])//2,
                (self.Camera_res[1]-self.Output_res[1])//2:(self.Camera_res[1]+self.Output_res[1])//2,
                :]
                results_mask = selfie_segmentation.process(image).segmentation_mask
                #results_mask = cv2.GaussianBlur(results_mask, (5, 5), 0)
                #切割子區域
                _, binary_mask = cv2.threshold(results_mask.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
                naz_y,naz_x = np.where(binary_mask) 
                crange = np.index_exp[np.min(naz_y):np.max(naz_y),np.min(naz_x):np.max(naz_x),:] #子區域範圍
                results_mask = cv2.cvtColor(results_mask, cv2.COLOR_GRAY2BGR) #1channel -> 3channel
                
                fg_image = np.multiply(image, results_mask).astype(np.uint8) #黑背景+全彩人像 float64 -> uint8
                fg_size = fg_image.shape[:2]
                #
                if len(self.frame_pos)==0:#底部置中
                    self.frame_pos=[
                        (self.Output_res[1] - fg_size[1])//2 ,
                        self.Output_res[0] - fg_size[0]
                    ]
                #背景
                bg_image = self.capvideo.read()
                bg_size  = bg_image.shape[:2]
                rate = max(self.Output_res[0]/bg_size[0] , self.Output_res[1]/bg_size[1])
                bg_image = cv2.resize(bg_image, (int(bg_size[1]*rate),int(bg_size[0]*rate)), interpolation=cv2.INTER_CUBIC)
                bg_size  = bg_image.shape[:2]
                bg_image = bg_image[
                (bg_size[0]-self.Output_res[0])//2:(bg_size[0]+self.Output_res[0])//2,
                (bg_size[1]-self.Output_res[1])//2:(bg_size[1]+self.Output_res[1])//2,
                :]
                bg_image[crange] = np.multiply(bg_image[crange], 1-results_mask[crange]).astype(np.uint8) #全彩背景+黑人像 float64 -> uint8
                #合成
                bg_image[crange] = bg_image[crange]+fg_image[crange]
                
                self.image=bg_image

if __name__=="__main__":
    b=backgoundchanger('bg.mp4')
    b.start()
    while True:
        #print(b.image)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the loop
        if key == ord("q"):
            break
        cv2.imshow("d",b.image)