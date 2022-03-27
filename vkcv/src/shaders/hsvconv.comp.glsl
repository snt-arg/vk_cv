#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba8) uniform readonly image2D inputImage;
layout(set = 0, binding = 1, rgba8) uniform image2D resultImage;

const vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
const float e = 1.0e-10;

vec4 rgb2hsv(in vec4 c) {
  vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
  vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
  float d = q.x - min(q.w, q.y);

  return vec4(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x, c.a);
}

void main() {
  ivec2 id = ivec2(gl_GlobalInvocationID.xy);

  vec4 rgb = imageLoad(inputImage, id);
  vec4 hsv = rgb2hsv(rgb);

  imageStore(resultImage, id, hsv);
}