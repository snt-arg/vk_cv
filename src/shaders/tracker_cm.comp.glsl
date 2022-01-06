#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout(set = 0, binding = 0, r8) uniform readonly image2D inputImage;
layout(set = 0, binding = 1, rgba32f) uniform image2D resultImage;

layout(constant_id = 3) const float inv_width = 1.0;
layout(constant_id = 4) const float inv_height = 1.0;

void main() {
  uvec2 id = gl_GlobalInvocationID.xy;
  uvec2 wgs = gl_WorkGroupSize.xy;

  // coordinate mask
  float r = imageLoad(inputImage, ivec2(id.xy)).r;
  vec2 d = r * vec2(id.xy) * vec2(inv_width, inv_height);

  imageStore(resultImage, ivec2(id.xy), vec4(d, r, 0));
}