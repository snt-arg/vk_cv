#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba8) uniform readonly image2D inputImage;
layout(set = 0, binding = 1, r8) uniform image2D resultImage;

void main() {
  ivec2 id = ivec2(gl_GlobalInvocationID.xy);

  vec4 rgb = imageLoad(inputImage, id);
  float r = (rgb.r + rgb.g + rgb.b) / 3.0;

  imageStore(resultImage, id, vec4(r));
}