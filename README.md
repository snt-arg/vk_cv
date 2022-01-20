# vk-cv

A fast and lightweight Vulkan accelerated graphics pipeline for simple color tracking.

## Pipeline

The pipeline consists of the following stages:

1. Acquisition (capture image from camera & upload to GPU)
2. HSV color-space conversion
3. Filtering based on hue
4. Morphological filtering
5. Tracking / centroid calculation 

![Alt text](media/pipeline.png?raw=true "Pipeline")

## Performance

On the Raspberry Pi4, the Vulkan accelerated pipeline is about 2x faster and takes 4x less CPU than the equivalent pipeline in OpenCV.

## Changes

### Version 0.1

* Initial release

## Requirements

Requires Vulkan 1.1 or higher.

## License

TBD