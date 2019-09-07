import cv2
import numpy as np
import time

'''
bg_img = cv2.imread(os.path.join(IMAGES_FOLDER, 'bg.jpg'))
current_frame_img = cv2.imread(os.path.join(IMAGES_FOLDER, 'current_frame.jpg'))

diff = cv2.absdiff(bg_img, current_frame_img)
mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
th, mask_thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

mask_indexes = mask_thresh > 0

foreground = np.zeros_like(current_frame_img, dtype=np.uint8)
for i, row in enumerate(mask_indexes):
    foreground[i, row] = current_frame_img[i, row]

plot_image(bg_img, recolour=True)
plot_image(current_frame_img, recolour=True)
plot_image(diff, recolour=True)
plot_image(mask)
plot_image(mask_thresh)
plot_image(foreground, recolour=True)
'''
bg_path = './dataset_maker/dataset/bg_img.png'  #640*480
data_path = './dataset_maker/dataset/'

width = 250
height = 250
red_color = (0,0,255)
blue_color = (255,0,0)
green_color = (0,255,0)


#bg_img = cv2.imread(bg_path, cv2.IMREAD_COLOR)
#bg_img = cv2.rectangle(bg_img, ((int)(640/2)-(int)(width/2),(int)(480/2)+(int)(height/2)), ((int)(640/2)+(int)(width/2),(int)(480/2)-(int)(height/2)), (0,0,255), 3)
#cv2.imshow('bg_img', bg_img)

##캡쳐cv2.imwrite("D:/" + str(now) + ".png", frame)

capture = cv2.VideoCapture(0)
print(capture.get(3),capture.get(4))

count = 0
delay = 8
delay += 1
file_name = 'ok'

print((int)(640/2)-(int)(width/2),(int)(480/2)+(int)(height/2), (int)(640/2)+(int)(width/2),(int)(480/2)-(int)(height/2))
'''
def mask(read_frame, bg, count):
    current_frame_img = cv2.imread("./dataset_maker/dataset/" + file_name + "/" + str(count) + ".png")
    bg_img = cv2.imread(bg, cv2.IMREAD_COLOR)
    print(read_frame)
    #current_frame_img = read_frame
    #current_frame_img = np.asarray(current_frame_img)

    diff = cv2.absdiff(bg_img, current_frame_img)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    th, mask_thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    mask_indexes = mask_thresh > 0

    foreground = np.zeros_like(current_frame_img, dtype=np.uint8)
    for i, row in enumerate(mask_indexes):
        foreground[i, row] = current_frame_img[i, row]
    
    plot_image(bg_img, recolour=True)
    plot_image(current_frame_img, recolour=True)
    plot_image(diff, recolour=True)
    plot_image(mask)
    plot_image(mask_thresh)
    plot_image(foreground, recolour=True)
    
    cv2.imwrite("./dataset_maker/dataset/" + file_name + "/" + str(count) + ".png", mask_thresh)
    #return mask_thresh
'''
def mask(read_frame, bg):
    current_frame = read_frame
    bg_img = cv2.imread(bg, cv2.IMREAD_COLOR)
    bg_img = np.asarray(bg_img)
    #print(read_frame)
    #current_frame_img = read_frame
    #current_frame_img = np.asarray(current_frame_img)

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

while count < 100 + delay-1:
    count += 1
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, ((int)(640/2)-(int)(width/2)-5,(int)(480/2)+(int)(height/2)+5), ((int)(640/2)+(int)(width/2)+5,(int)(480/2)-(int)(height/2)-5), (0,0,255), 3)
    cv2.imshow('VideoFrame', frame)
    #cv2.imwrite("./dataset_maker/dataset/" + file_name + str(count) + ".png", frame)
    if count < delay : 
        if cv2.waitKey(1)>0:break
        print('no capture its delay time, {}'.format(count))
        #print(count)
        time.sleep(1)
        continue
    time.sleep(0.5)
    #mask(frame, bg_path)
    im_trim(frame, count-delay+1)
    if cv2.waitKey(1)>0:break

capture.release()
#cv2.waitKey(0)
cv2.destroyAllWindows()
