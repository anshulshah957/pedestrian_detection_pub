# pedestrian_detection

We use the [YOLO](https://pjreddie.com/darknet/yolo/) framework to classify pedestrians, vehicles, traffic lights and other basic objects. In addition, we also implement other algorithms for lane detection and to further classify traffic lights as red, green, or yellow.

We apply our object detection algorithms to implement a "self-driving" car in GTAV that can detect the above objects and make simple decisions.

## Dependencies
* [YOLO](https://pjreddie.com/darknet/yolo/)
* [darkflow](https://github.com/thtrieu/darkflow)

### darkflow
darkflow transfers darknet (and YOLO) to tensorflow.

Run the following python command in the darkflow directory:
`flow --model cfg/yolo.cfg --load yolo.weights --demo testvid.mp4 --saveVideo`

#### Tags
`--model` different configurations are available in /cfg
`--load` different weights are available on [Darkflow](https://github.com/thtrieu/darkflow)
`--demo` specifies the input video
`--saveVideo` saves the output video as video.avi (you might have to use a video converter like media.io to open)
