#version 450

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, r8) uniform readonly image2D inputImage;
layout(set = 0, binding = 1, r8) uniform image2D resultImage;

// row major
layout(constant_id = 2) const float m11 = 1.0;
layout(constant_id = 3) const float m12 = 1.0;
layout(constant_id = 4) const float m13 = 1.0;
layout(constant_id = 5) const float m21 = 1.0;
layout(constant_id = 6) const float m22 = 1.0;
layout(constant_id = 7) const float m23 = 1.0;
layout(constant_id = 8) const float m31 = 1.0;
layout(constant_id = 9) const float m32 = 1.0;
layout(constant_id = 10) const float m33 = 1.0;

layout(constant_id = 11) const int erode_dilate = 1;

float op(in float a, in float b) {
  if (erode_dilate == 0) {
    return min(a, b);
  } else {
    return max(a, b);
  }
}

void main() {
  uvec2 id = gl_GlobalInvocationID.xy;

  float avg[9];
  avg[0] = m11 * imageLoad(inputImage, ivec2(id.x - 1, id.y - 1)).r;
  avg[1] = m12 * imageLoad(inputImage, ivec2(id.x - 1, id.y - 0)).r;
  avg[2] = m13 * imageLoad(inputImage, ivec2(id.x - 1, id.y + 1)).r;
  avg[3] = m21 * imageLoad(inputImage, ivec2(id.x - 0, id.y - 1)).r;
  avg[4] = m22 * imageLoad(inputImage, ivec2(id.x - 0, id.y - 0)).r;
  avg[5] = m23 * imageLoad(inputImage, ivec2(id.x - 0, id.y + 1)).r;
  avg[6] = m31 * imageLoad(inputImage, ivec2(id.x + 1, id.y - 1)).r;
  avg[7] = m32 * imageLoad(inputImage, ivec2(id.x + 1, id.y - 0)).r;
  avg[8] = m33 * imageLoad(inputImage, ivec2(id.x + 1, id.y + 1)).r;

  float res = op(avg[0], avg[1]);
  res = op(res, avg[2]);
  res = op(res, avg[3]);
  res = op(res, avg[4]);
  res = op(res, avg[5]);
  res = op(res, avg[6]);
  res = op(res, avg[7]);
  res = op(res, avg[8]);

  imageStore(resultImage, ivec2(id.xy), vec4(res));
}