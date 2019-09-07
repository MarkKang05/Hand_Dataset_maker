import cv2
import numpy as np
import time

bg_path = './dataset_maker/dataset/bg_img.png'  #640*480
data_path = './dataset_maker/dataset/'

width = 250 #guide box's width
height = 250 #guid box's height

capture = cv2.VideoCapture(0)
print(capture.get(3),capture.get(4))

count = 0
delay = 8 # The first time the program runs, it will give you 8 test times.
delay += 1
file_name = 'ok' #insert data folder name

def mask(read_frame, bg):
    current_frame = read_frame
    bg_img = cv2.imread(bg, cv2.IMREAD_COLOR)
    bg_img = np.asarray(bg_img)
    diff = cv2.absdiff(bg_img, current_frame)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    th, mask_thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    return mask_thresh


def im_trim(frame,count):
    
    #crop_img = img[y:h, x:w] # Crop from x, y, w, h -> 100, 200, 300, 400
    img_trim = frame[115:115+250, 195:195+250]
    im_trim = np.asarray(img_trim)
    cv2.imwrite("./dataset_maker/dataset/" + file_name + "/" + str(count) + ".png", im_trim)

    current_img = cv2.imread("./dataset_maker/dataset/" + file_name + "/" + str(count) + ".png")
    bg_img = cv2.imread(bg_path, cv2.IMREAD_COLOR)

    diff = cv2.absdiff(bg_img, current_img)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    th, mask_thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    cv2.imwrite("./dataset_maker/dataset/" + file_name + "/" + str(count) + ".png", mask_thresh)

    print("capture {}".format(count))

while count < 100 + delay-1: # it will store 100 photos in total.
    count += 1
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    # 640, 480 of bottom line are my camera's resolution.
    cv2.rectangle(frame, ((int)(640/2)-(int)(width/2)-5,(int)(480/2)+(int)(height/2)+5), ((int)(640/2)+(int)(width/2)+5,(int)(480/2)-(int)(height/2)-5), (0,0,255), 3)
    cv2.imshow('VideoFrame', frame)
    
    if count < delay : 
        if cv2.waitKey(1)>0:break
        print('no capture its delay time, {}'.format(count))
        #print(count)
        time.sleep(1)
        continue
    time.sleep(0.5)
    im_trim(frame, count-delay+1)
    if cv2.waitKey(1)>0:break

capture.release()
#cv2.waitKey(0)
cv2.destroyAllWindows()
