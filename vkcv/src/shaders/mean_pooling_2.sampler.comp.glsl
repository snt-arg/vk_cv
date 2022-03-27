#version 450

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) uniform sampler2D inputImageSampler; // linear
layout(set = 0, binding = 1, rgba32f) uniform image2D resultImage;

layout(constant_id = 2) const float inv_size = 1.0; // of the output image

void main() {
  ivec2 id = ivec2(gl_GlobalInvocationID.xy);

  // scale down by a factor of two by sampling between the
  // texels, thus getting the average of the 4 neighbouring
  // texels. The idea is to use the specialized hardware to
  // perform the filtering for us.
  //
  // [0,0]---[1,0]
  //   |   x   |     sample location (0.5;0.5)
  // [0,1]---[1,1]

  // normalized texture coords
  vec2 uv = (vec2(id) + vec2(0.5)) * inv_size;
  vec4 d = texture(inputImageSampler, uv);

  imageStore(resultImage, id, d);
}
