#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba8) uniform readonly image2D inputImage;
layout(set = 0, binding = 1, rgba8) uniform image2D resultImage;

// layout(constant_id = 0) const float[9] kernel = float[](-1.0,  0.0, 0.0,
//     0.0, -1.0, 0.0,
//     0.0,  0.0, 2.0);

// 128 bytes max.
layout(push_constant) uniform PushConstants {
  float[9] kernel;
  float offset;
  float denom;
}
pc;

float conv(in float[9] kernel, in float[9] data, in float denom,
           in float offset) {
  float acc = 0.0;
  for (int i = 0; i < 9; ++i) {
    acc += kernel[i] * data[i];
  }
  return clamp(acc / denom + offset, 0.0, 1.0);
}

vec3 rgb2hsv(vec3 c) {
  vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
  vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

  float d = q.x - min(q.w, q.y);
  float e = 1.0e-10;
  return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// const float[9] kernel = float[](-1.0,  0.0, 0.0,
//                                  0.0, -1.0, 0.0,
//                                  0.0,  0.0, 2.0);

void main() {
  uvec2 idx = gl_GlobalInvocationID.xy;

  float avg[9];

  int n = -1;
  for (int i = -1; i < 2; ++i) {
    for (int j = -1; j < 2; ++j) {
      n++;
      vec3 rgb = imageLoad(inputImage, ivec2(idx.x + i, idx.y + j)).rgb;
      avg[n] = (rgb.r + rgb.g + rgb.b) / 3.0;
    }
  }

  vec4 res = vec4(vec3(conv(pc.kernel, avg, pc.offset, pc.denom)), 1.0);

  imageStore(resultImage, ivec2(idx.xy), res);
}