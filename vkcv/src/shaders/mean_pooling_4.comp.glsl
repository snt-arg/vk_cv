#version 450

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba32f) uniform image2D inputImage;
layout(set = 0, binding = 1, rgba32f) uniform image2D resultImage;

void main() {
  ivec2 id = ivec2(gl_GlobalInvocationID.xy);

  // Note: The RPi misbehaves when using the sampler method
  //       However, this approach works.
  //       Furthermore, it has no performance penalty on the RPi.
  ivec2 p = id * 4;
  vec4 d = imageLoad(inputImage, p + ivec2(0, 0));
  d += imageLoad(inputImage, p + ivec2(0, 1));
  d += imageLoad(inputImage, p + ivec2(0, 2));
  d += imageLoad(inputImage, p + ivec2(0, 3));

  d += imageLoad(inputImage, p + ivec2(1, 0));
  d += imageLoad(inputImage, p + ivec2(1, 1));
  d += imageLoad(inputImage, p + ivec2(1, 2));
  d += imageLoad(inputImage, p + ivec2(1, 3));

  d += imageLoad(inputImage, p + ivec2(2, 0));
  d += imageLoad(inputImage, p + ivec2(2, 1));
  d += imageLoad(inputImage, p + ivec2(2, 2));
  d += imageLoad(inputImage, p + ivec2(2, 3));

  d += imageLoad(inputImage, p + ivec2(3, 0));
  d += imageLoad(inputImage, p + ivec2(3, 1));
  d += imageLoad(inputImage, p + ivec2(3, 2));
  d += imageLoad(inputImage, p + ivec2(3, 3));

  imageStore(resultImage, id, d * 0.0625);
}