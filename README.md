# pedestrian_detection

We use the darknet framework to classify pedestrians, vehicles, traffic lights and other basic objects. In addition, we also implement other algorithms for lane detection and to further classify traffic lights as red, green, or yellow.

We apply our object detection algorithms to implement a "self-driving" car in GTAV that can detect the above objects and make simple decisions.

## Dependencies
* Python3
* OpenCV
* NumPy
* pynput

### [darknet](https://pjreddie.com/darknet/yolo/)
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
