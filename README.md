# pedestrian_detection

We use the YOLO Algorithm to classify pedestrians, vehicles, and other basic objects. In addition, we also implement other algorithms for lane detection and to classify a  traffic light intersection as either red, green, or yellow.

We are in the process of integrating these detection algorithms to implement a "self driving car" in the DeepDrive simulator as well as in GTA V that can detect the above  objects and make simple decisions

![alt text](https://raw.githubusercontent.com/CS196Illinois/pedestrian_detection/master/yolo_screenshot.png)

## Dependencies
* GTAV
* DeepDrive
* Python3
* OpenCV
* NumPy
* pynput  
* Pillow  
* mss

### [darknet](https://pjreddie.com/darknet/yolo/)
`cd vision/object_detection/darknet`  
`make`  
`wget https://pjreddie.com/media/files/yolov3.weights`
## Demos
### Darknet
Note: our configuration currently only supports numpy arrays.
#### Image
Writes bounding box data to output.txt.
* cd to pedestrian_detection directory  
* open test_image.py  
* change path to image file on line 16  
* `python test_image.py`

#### Video
* cd to pedestrian_detection directory  
* open test_video.py  
* change path to video file on line 15  
* `python test_video.py`

Note: [compile with CUDA](https://pjreddie.com/darknet/install/#cuda) for 500x speedup.

### Traffic Light Detection
* 'cd vision/traffic_light'
* open 'trafficLight.py'
* add filename you want to test in line 65
* 'python3 trafficLight.py'

## Authors
Ananmay Jain (project manager)  
Tim Baer  
Thomas Mao  
Anshul Shah

## License
This project is distributed under the GNU General Public Liscense - see the LICENSE.md file for more details

## Acknowledgments
Title: YOLOV3: An Incremental Improvement  
Author: Redmon, Joseph, and Farhadi, Ali  
Journal: arXiv  
Year: 2018  
Availability: https://pjreddie.com/darknet/yolo/  

Title: Not just another YOLO V3 for Python (comment)  
Author: Glenn Jocher  
Year: 2018  
Availability: https://medium.com/@glenn.jocher/to-follow-up-i-updated-array-to-image-and-detect-to-utilize-numpy-edf326171e76

Title: Vehicle Detection for Autonomous Driving  
Author: Junsheng Fu  
Year: 2018  
Availability: https://github.com/JunshengFu/vehicle-detection

