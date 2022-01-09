#version 450

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) uniform sampler2D inputImageSampler; // linear
layout(set = 0, binding = 1, rgba32f) uniform image2D resultImage;

layout(constant_id = 2) const float inv_size = 1.0;

void main() {
  ivec2 id = ivec2(gl_GlobalInvocationID.xy);

  // the coordinate in the bigger (input) picture
  vec2 coord = vec2(id * 2) * inv_size;

  // scale down by a factor of 2
  const vec2 dd = vec2(0.5, 0.5) * inv_size;
  vec3 d = texture(inputImageSampler, coord + dd).rgb;

  imageStore(resultImage, id, vec4(d, 0));
}
