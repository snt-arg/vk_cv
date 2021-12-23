#version 450

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, r8) uniform readonly image2D inputImage;
layout(set = 0, binding = 1, r8) uniform image2D resultImage;

layout(constant_id = 2) const float m1 = -1.0;
layout(constant_id = 3) const float m2 = 0.0;
layout(constant_id = 4) const float m3 = 0.0;
layout(constant_id = 5) const float m4 = 0.0;
layout(constant_id = 6) const float m5 = -1.0;
layout(constant_id = 7) const float offset = 0.5;
layout(constant_id = 8) const float denom = 2.0;
layout(constant_id = 9) const int v_pass = 0;

float conv(in float[5] data) {
  float acc = m1 * data[0] + m2 * data[1] + m3 * data[2] //
              + m4 * data[3] + m5 * data[4];

  return clamp(acc / denom + offset, 0.0, 1.0);
}

void main() {
  /*
  Note:
  - The compiler doesn't seem to unroll the loops
  - Shared memory doesn't seem to improve performance
  - Using push constants does not impact performance
    (though does not optimize away 0 elements)
  */
  uvec2 id = gl_GlobalInvocationID.xy;
  uvec2 lid = gl_LocalInvocationID.xy;
  uvec2 wgid = gl_WorkGroupID.xy;

  float avg[5];
  if (v_pass == 0) {
    avg[0] = imageLoad(inputImage, ivec2(id.x - 2, id.y)).r;
    avg[1] = imageLoad(inputImage, ivec2(id.x - 1, id.y)).r;
    avg[2] = imageLoad(inputImage, ivec2(id.x - 0, id.y)).r;
    avg[3] = imageLoad(inputImage, ivec2(id.x + 1, id.y)).r;
    avg[4] = imageLoad(inputImage, ivec2(id.x + 2, id.y)).r;
  } else {
    avg[0] = imageLoad(inputImage, ivec2(id.x, id.y - 2)).r;
    avg[1] = imageLoad(inputImage, ivec2(id.x, id.y - 1)).r;
    avg[2] = imageLoad(inputImage, ivec2(id.x, id.y - 0)).r;
    avg[3] = imageLoad(inputImage, ivec2(id.x, id.y + 1)).r;
    avg[4] = imageLoad(inputImage, ivec2(id.x, id.y + 2)).r;
  }

  vec4 res = vec4(conv(avg));

  imageStore(resultImage, ivec2(id.xy), res);
}