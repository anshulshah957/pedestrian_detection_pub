# pedestrian_detection

We use the YOLO Algorithm to classify pedestrians, vehicles, and other basic objects. In addition, we also implement other algorithms for lane detection and to classify a traffic light intersection as either red, green, or yellow.

We are in the process of integrating these detection algorithms to implement a "self driving car" in the DeepDrive simulator as well as in GTA V that can detect the above objects and make simple decisions

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
* `cd vision/object_detection/darknet`  
`make`  
* download [pre-trained weight file](https://pjreddie.com/media/files/yolov3.weights) to darknet directory

## Authors
Ananmay Jain (project manager)  
Tim Baer  
Thomas Mao  
Anshul Shah

## License
This project is distributed under the MIT license - see the LICENSE.md file for more details

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

