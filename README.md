# pedestrian_detection

We use the YOLO framework to classify pedestrians, vehicles, traffic lights and other basic objects. In addition, we also implement other algorithms for lane detection and to further classify traffic lights as red, green, or yellow.

We apply our object detection algorithms to implement a "self-driving" car in GTAV that can detect the above objects and make simple decisions.

## Dependencies
### [YOLO](https://pjreddie.com/darknet/yolo/)
* ???

### [darkflow](https://github.com/thtrieu/darkflow)
* Python3
* Tensorflow (requires Python 3.4, 3.5, or 3.6)
* NumPy
* OpenCV
* Cython

### darkflow
darkflow transfers darknet (and YOLO) to tensorflow.

Download weights [here](https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU) and move to darkflow directory.

Run the following python command in the darkflow directory:
`flow --model cfg/yolo.cfg --load yolo.weights --demo testvid.mp4 --saveVideo`

#### Tags
`--model` different configurations are available in /cfg  
`--load` different weights are available on darkflow GitHub  
`--demo` specifies the input video  
`--saveVideo` saves the output video as video.avi (you might have to use a video converter like media.io to open)  

## Authors
Tim Baer  
Ananmay Jain (project manager)  
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

Title: darkflow  
Author: thtrieu and other contributors  
Year: 2018  
Availability: https://github.com/thtrieu/darkflow  

