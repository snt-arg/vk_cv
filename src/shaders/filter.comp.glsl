#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba8) uniform readonly image2D inputImage;
layout(set = 0, binding = 1, r8) uniform image2D resultImage;

layout(push_constant) uniform PushConstants {
  vec3 rgb_max;
  vec3 rgb_min;
}
pc;

void main() {
  uvec2 id = gl_GlobalInvocationID.xy;

  vec3 rgb = imageLoad(inputImage, ivec2(id.x, id.y)).rgb;
  float v = 0;

  if (all(greaterThan(rgb, pc.rgb_min)) && all(lessThan(rgb, pc.rgb_max))) {
    v = 1.0;
  }

  imageStore(resultImage, ivec2(id.xy), vec4(v));
}