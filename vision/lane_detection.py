import cv2
import numpy as np

# detect edges (sudden change in pixel values) through
# canny edge dection
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
    
    # fill the mask with white in the polygon region and black everywhere else 
    cv2.fillPoly(mask, np.array([arr_ver], dtype=np.int32), mask_color)
    
    # reduces pixel value to zero wherever mask is zero
    return cv2.bitwise_and(img,mask)

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
            minLineLength = 40,
            maxLineGap = 25
            )

def draw_lines(img, lines, color = [255, 0, 0], thickness = 3, slope_threshold = 0.5):
    if lines == None:
        return
    
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    line_image = np.copy(img)

    for line in lines:
        for x0, y0, x1, x2 in line:
            slope = (y1 - y0)/(x1 - x0)
            if abs(slope) < slope_threshold:
                continue
            if slope < 0:
                left_line_x.append([x0,x1])
                left_line_y.append([y0,y1])
            else:
                right_line_x.append([x0,x1])
                right_line_y.append([y0,y1])
    
    left_line = np.polyfit(left_line_y, left_line_x, 1)
    right_line = np.polyfit(right_line_y, right_line_x, 1)
    
    poly_left = np.poly1d(left_line)
    poly_right = np.poly1d(right_line)
    
    # just below our triangular cropped image
    min_y = img.shape[0] * (3/5)
    max_y = img.shape[0]

    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))

    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

    cv2.line(line_image, (left_x_start,left_x_end), (max_y, min_y), thickness)
    cv2.line(line_image, (right_x_start,right_x_end), (max_y, min_y), thickness)

    return line_image

def main():
    vid = cv2.VideoCapture('clip_highway_video.mp4')

    while vid.isOpened():
        
        ret, frame = vid.read()
        print("image read")
        gray = grayscale(frame)
        print("grayscale works")

        canny_image = canny_edge(gray)
        print("canny works")
        cv2.imshow('canny', canny_image)

        width = canny_image.shape[1]
        height = canny_image.shape[0]
        region = [(0,0), (int(width/2),int(height/2)), (width,height)]
        mask_image = image_mask(canny_image, region)
        print("mask works")
        #cv2.imshow('mask', mask_image)

        lines = hough_transform(mask_image)
        print("hough works")

        line_image = draw_lines(frame, lines)
        print("draw lines works")

        #cv2.imshow('line_image',line_image)
        cv2.waitKey(1)

    vid.release()
    cv2.destrolAllWindows()

if __name__ == '__main__':
    main()
