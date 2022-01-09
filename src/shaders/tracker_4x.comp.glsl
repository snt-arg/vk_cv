#version 450

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) uniform sampler2D inputImageSampler; // linear
layout(set = 0, binding = 1, rgba32f) uniform image2D resultImage;

layout(constant_id = 2) const float inv_size = 1.0;

void main() {
  ivec2 id = ivec2(gl_GlobalInvocationID.xy);

  // the coordinate in the bigger (input) picture
  vec2 coord = vec2(id * 4) * inv_size;

  const vec2 dp0 = vec2(0.5, 0.5) * inv_size;
  const vec2 dp1 = vec2(1.5, 0.5) * inv_size;
  const vec2 dp2 = vec2(0.5, 1.5) * inv_size;
  const vec2 dp3 = vec2(1.5, 1.5) * inv_size;

  // scale down by a factor of 4
  vec3 p0 = texture(inputImageSampler, coord + dp0).rgb;
  vec3 p1 = texture(inputImageSampler, coord + dp1).rgb;
  vec3 p2 = texture(inputImageSampler, coord + dp2).rgb;
  vec3 p3 = texture(inputImageSampler, coord + dp3).rgb;

  vec3 d = (p0 + p1 + p2 + p3) * 0.25;

  imageStore(resultImage, id.xy, vec4(d, 0));
}