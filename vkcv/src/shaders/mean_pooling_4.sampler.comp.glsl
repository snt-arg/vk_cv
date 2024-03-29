#version 450

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) uniform sampler2D inputImageSampler; // linear
layout(set = 0, binding = 1, rgba16f) uniform image2D resultImage;

layout(constant_id = 2) const float inv_size = 1.0;

const vec2 dp0 = vec2(0.5, 0.5);
const vec2 dp1 = vec2(2.5, 0.5);
const vec2 dp2 = vec2(0.5, 2.5);
const vec2 dp3 = vec2(2.5, 2.5);

void main() {
  ivec2 id = ivec2(gl_GlobalInvocationID.xy);

  // scale down by a factor of two by sampling between the
  // texels, thus getting the average of the 4 neighbouring
  // texels which we then average.
  //
  // [0,0]---[1,0]---[2,0]---[3,0]
  //   |   x   |       |   x   |      sample locations: (0.5;0.5), (2.5;0.5)
  // [0,1]---[1,1]---[2,1]---[3,1]
  //   |       |       |       |
  // [0,2]---[1,3]---[2,3]---[3,3]
  //   |   x   |       |   x   |      sample locations: (0.5;2.5), (2.5;2.5)
  // [0,4]---[1,4]---[2,4]---[3,4]

  // scale down by a factor of 4
  vec4 d = texture(inputImageSampler, (id + dp0) * inv_size);
  d += texture(inputImageSampler, (id + dp1) * inv_size);
  d += texture(inputImageSampler, (id + dp2) * inv_size);
  d += texture(inputImageSampler, (id + dp3) * inv_size);

  imageStore(resultImage, id, d * 0.25);
}