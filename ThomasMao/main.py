from moviepy.editor import VideoFileClip
from svm_pipeline import *
from yolo_pipeline import *
from lane import *
import cv2
import imageio
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2
from skimage.feature import hog
from peddetector import detectped_BGRinput
import time
from PIL import ImageGrab
#from trafficLightDetector import detectcircle
def pipeline_yolo(img):

    img_undist, img_lane_augmented, lane_info = lane_process(img)
    output = vehicle_detection_yolo(img_undist, img_lane_augmented, lane_info)

    return output

def pipeline_svm(img):

    img_undist, img_lane_augmented, lane_info = lane_process(img)
    output = vehicle_detection_svm(img_undist, img_lane_augmented, lane_info)

    return output
def canny_edge(img, low_thresh = 100, up_thresh = 200):
    return cv2.Canny(img, low_thresh, up_thresh)

# convert image to grayscale
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# image cropper
def image_mask(img, arr_ver):

    # image mask
    mask = np.zeros_like(img)
    # Checking no. of color channels
    if len(img.shape) > 2:
        channels = img.shape[2]
        mask_color = (255,) * channels
    else:
        mask_color = 255

    #  arr = np.array(arr_ver, dtype=np.int32)
    # print(arr)

    width = img.shape[1]
    height = img.shape[0]

    left_line = np.polyfit([height, height/2], [0,width/2], 1)
    right_line = np.polyfit([height,height/2], [width, width/2], 1)

    fl0 = np.poly1d(left_line)
    fl1 = np.poly1d(right_line)

    for j in range(int(height/2), int(height)):
        for i in range(int(fl0(j)),int(fl1(j))):
            mask[j][i] = mask_color

    # fill the mask with black everywhere else except the region
    # cv2.fillPoly(mask, np.int_([arr]), color =  mask_color)
    # reduces pixel value to zero wherever mask is zero

    # img with canny edges only in bottom screen triangle
    return_img =  cv2.bitwise_and(img,mask)

    # cv2.imshow('img', return_img)
    return return_img

def plot_region(img):
    width = img.shape[1]
    height = img.shape[0]
    region = [(0,0), (width/2,height/2), (width,height)]
    masked_image = image_mask(img, region)

    cv2.imshow('image', masked_image)

def hough_transform(img):
    return cv2.HoughLinesP(
            img,
            rho = 6,
            theta = np.pi/60,
            threshold = 160,
            lines = np.array([]),
                minLineLength = 100,
            maxLineGap = 60
            )

def draw_lines(img, lines, color = [0, 255, 0], thickness = 3, slope_threshold = 0.5):
    if len(lines) == 0:
        return

    left_line = []
    right_line = []

    line_image = np.copy(img)
    cv2.imshow('img',img)
    for i in range (len(lines)):
        x0 = lines[i][0][0]
        y0 = lines[i][0][1]
        x1 = lines[i][0][2]
        y1 = lines[i][0][3]
        slope = (y1 - y0)/(x1 - x0)
        if abs(slope) < slope_threshold:
            continue
        if slope < 0:
            left_line_x = [x0,x1]
            left_line_y = [y0,y1]
            # if len(left_line_y) != 0 and len(left_line_x) != 0:
            left_line.append(np.polyfit(left_line_y, left_line_x, 1))
        else:
            right_line_x = [x0,x1]
            right_line_y = [y0,y1]
            # if len(right_line_y) != 0 and len(right_line_x) != 0:
            right_line.append(np.polyfit(right_line_y, right_line_x, 1))

    sum_l_m = 0
    sum_l_b = 0
    sum_r_m = 0
    sum_r_b = 0

    for i in range(len(left_line)):
        sum_l_m += left_line[i][0]
        sum_l_b += left_line[i][1]
    for i in range(len(right_line)):
        sum_r_m += right_line[i][0]
        sum_r_b += right_line[i][1]

    left_m = 0
    left_b = 0
    right_m = 0
    right_b = 0

    if len(left_line) != 0:
        left_m = sum_l_m/len(left_line)
        left_b = sum_l_b/len(left_line)

    if len(right_line) != 0:
        right_m = sum_r_m/len(right_line)
        right_b = sum_r_b/len(right_line)

    poly_left = np.poly1d([left_m, left_b])
    poly_right = np.poly1d([right_m, right_b])

    print(poly_left)
    print(poly_right)
    # just below our triangular cropped image
    min_y = int(img.shape[0] * (3/5))
    max_y = int(img.shape[0])

    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))

    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

    mask = np.zeros_like(img)

    cv2.line(img, (left_x_start,max_y), (left_x_end, min_y), color, thickness)
    cv2.line(img, (right_x_start,max_y), (right_x_end, min_y), color, thickness)

    return img




if __name__ == "__main__":

    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('deepdrive1.mp4',fourcc, 20.0, (1280,720))
    for i in range(5):
        print(i+1)
        time.sleep(1)
    while True:
        screen =  np.array(ImageGrab.grab())
        screen = cv2.resize(screen, (1280,720))
        screen = pipeline_yolo(screen)
        
        #main(img, net, meta)
        #cap = cv2.VideoCapture('pov.mp4')
	    #while(cap.isOpened()):
		#ret, frame = cap.read()
        
        #imageout=pipeline_yolo(cv2.resize(screen,(1280,720)))
        #cv2.imshow('frame',lane_detection_image)
        screen =cv2.cvtColor(screen,cv2.COLOR_RGB2BGR)
        cv2.imshow('screen',screen)
        out.write(screen)
        #move(False, False, False, poly_left, poly_right,img.shape[0], img.shape[1])
		# trafficlights = traffic_data(frame);
        # print(trafficlights)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    out.release()
    cv2.destroyAllWindows()
                
        
        
        
        
        
        
        


   


