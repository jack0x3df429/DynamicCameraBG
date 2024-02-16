################################
import pyvirtualcam
from backgoundchanger import backgoundchanger
################################
import cv2,sys
################################

################################
fmt = pyvirtualcam.PixelFormat.BGR
################################
Screen_res=(900,1600)
Camera_res=(720,1280)
Output_res=(720,1280)
################################

with pyvirtualcam.Camera(width=Output_res[1], height=Output_res[0], fps=30, fmt=fmt) as cam:
    bgcg=backgoundchanger(sys.argv[1],Output_res)
    #capvideo=bgcg.capvideo
    bgcg.start()
    while True:
        out=bgcg.image
        #print(out.shape)
        #out=cv2.resize(out, (Output_res[0],Output_res[1]), interpolation=cv2.BORDER_DEFAULT)
        cam.send(out.astype("uint8"))
        cv2.imshow("my webcam", out)
        cam.sleep_until_next_frame()
        key = cv2.waitKey(1) & 0xFF
        # Q鍵退出
        if key == ord("q"):
            bgcg.stop_flag=True
            bgcg.join()
            break
    cv2.destroyAllWindows()
    #break