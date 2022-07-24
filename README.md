# vk-cv

A fast and lightweight Vulkan (GPU) accelerated computer vision pipeline for simple color tracking.

## Pipeline

The pipeline consists of the following stages:

1. Acquisition (capture image from camera & upload to GPU)
2. HSV color-space conversion
3. Filtering based on hue
4. Morphological filtering (3x3)
5. Max pooling (2x2)
6. Tracking / centroid calculation 

![Alt text](vkcv/media/pipeline.png?raw=true "Pipeline")

## Performance

On the Raspberry Pi4, the Vulkan accelerated pipeline is about 2x faster and takes 4x less CPU than the equivalent pipeline in OpenCV.

## ROS Topics


`~/local_point`: `geometry_msgs::msg::Point` the 3D position (in the camera frame) of the detected color blob.

`~/camera_image/compressed`: `sensor_msgs::msg::CompressedImage` the compressed camera image including the crosshair pointing at the detected color blob.

`~/lock`: `std_msgs::msg::Bool` indicates whether a color blob is currently detected.

## Build Instructions

Make sure `libturbojpeg` is installed and your ROS environment is sourced (`source source /opt/ros/<dist>/setup.bash`).

Build with
```
cargo b --release
```
the compiled executable is placed in `./target/release/ros2-publisher`.

```
ros1-publisher 1.0.0

USAGE:
    ros1-publisher [FLAGS] [OPTIONS] [__ros_args]...

FLAGS:
    -h, --help                    Prints help information
        --raw-color-image         Transmit unprocessed (yet compressed) color image
        --transmit-depth-image    Transmit depth image
    -t, --transmit-image          Transmits the camera image with a crosshair. Images are compressed via libjpegturbo.
                                  WARNING: This may generate a lot of data!
    -V, --version                 Prints version information
    -v, --verbose                 Be verbose

OPTIONS:
    -c, --compressor-quality <compressor-quality>    Compression quality [default: 60]
    -l, --lock-timeout <lock-timeout>                Lock timeout in ms [default: 1000]
    -m, --min-area <min-area>
            The smallest area in pixels required by the detector. Smaller areas will be ignored [default: 110]


ARGS:
    <__ros_args>...    Roslaunch adds some special args e.g. __name:=... __log:=... [default: ]
```

## Changes

### Version 0.4
* vulkano 0.30
* ROS1 gazebo node
* OpenCV reference
* histograms

### Version 0.3

* Improved: ROS1 & ROS2 integration
* Added: Hole fill filter
* Added: Transmit compressed depth image

### Version 0.2

* Added: ROS2 integration and compressed camera image streaming
* Added: Max pooling before tracker

### Version 0.1

* Initial release

## Requirements

Requires Vulkan 1.1 or higher.

## License

TBD