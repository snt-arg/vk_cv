#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba32f) uniform readonly image2D inputImage;
layout(set = 0, binding = 1, rgba32f) uniform image2D resultImage;

void main() {
  uvec2 id = gl_GlobalInvocationID.xy;
  uvec2 wgs = gl_WorkGroupSize.xy;

  // the coordinate in the bigger (input) picture
  ivec2 coord = ivec2(id) * 2;

  // scale down by a factor of 2
  vec3 p1 = imageLoad(inputImage, coord + ivec2(0, 0)).rgb;
  vec3 p2 = imageLoad(inputImage, coord + ivec2(1, 0)).rgb;
  vec3 p3 = imageLoad(inputImage, coord + ivec2(0, 1)).rgb;
  vec3 p4 = imageLoad(inputImage, coord + ivec2(1, 1)).rgb;

  vec3 d = (p1 + p2 + p3 + p4) * 0.25;

  imageStore(resultImage, ivec2(id.xy), vec4(d, 0));
}