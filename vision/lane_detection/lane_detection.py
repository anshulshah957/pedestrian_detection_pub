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
            threshold = 600,
            lines = np.array([]),
                minLineLength = 100,
            maxLineGap = 60
            )

def draw_lines(img, lines, color = [0, 255, 0], thickness = 3, slope_threshold = 0.5):
    if lines == None:
        return
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

    # print(poly_left)
    # print(poly_right)
    # # just below our triangular cropped image
    # min_y = int(img.shape[0] * (3/5))
    # max_y = int(img.shape[0])

    # left_x_start = int(poly_left(max_y))
    # left_x_end = int(poly_left(min_y))

    # right_x_start = int(poly_right(max_y))
    # right_x_end = int(poly_right(min_y))

    # mask = np.zeros_like(img)

    # cv2.line(img, (left_x_start,max_y), (left_x_end, min_y), color, thickness)
    # cv2.line(img, (right_x_start,max_y), (right_x_end, min_y), color, thickness)

    return poly_left, poly_right

def intersect_lines(poly_left, poly_right):
    left_b = poly_left[0]
    left_m = poly_left[1]
    right_b = poly_right[0]
    right_m = poly_right[1]
    
    a = np.array([[-left_m, 1], [-right_m, 1]])
    b = np.array([left_b, right_b])

    # returns [x y]
    return np.linalg.solve(a, b)

def main(frame):
    # vid = cv2.VideoCapture('clip_highway_video.mp4')
    # frame_width = int(vid.get(3))
    # frame_height = int(vid.get(4))
    # out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))
    # while vid.isOpened():

        # ret, frame = vid.read()

    gray = grayscale(frame)

    canny_image = canny_edge(gray)

    width = canny_image.shape[1]
    height = canny_image.shape[0]

    region = [(0,0), (int(width/2),int(height/2)), (width,height)]
    mask_image = image_mask(canny_image, region)

    lines = hough_transform(mask_image)
        # for i in range(len(lines)):
        #     cv2.line(frame,(lines[i][0][0], lines[i][0][1]),(lines[i][0][2], \
        #             lines[i][0][3]),(0,255,0), 3)
        # for x1,y1,x2,y2 in lines[0]:
        #     cv2.line(mask_image,(x1,y1),(x2,y2),(0,255,0),10)
        # line_image = draw_lines(mask_image,lines)
        
        # line_image, poly_left, poly_right = draw_lines(frame, lines)
    return draw_lines(frame, lines)
        
        # cv2.waitKey(1)

    # vid.release()
    # out.release()
    # cv2.destrolAllWindows()




'''if __name__ == '__main__':
    main()'''
