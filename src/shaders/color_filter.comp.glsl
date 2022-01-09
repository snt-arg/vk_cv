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
  ivec2 id = ivec2(gl_GlobalInvocationID.xy);

  vec3 rgb = imageLoad(inputImage, id).rgb;

  float v = 0;
  if (all(greaterThanEqual(rgb, pc.rgb_min)) &&
      all(lessThanEqual(rgb, pc.rgb_max))) {
    v = 1.0;
  }

  imageStore(resultImage, id, vec4(v));
}