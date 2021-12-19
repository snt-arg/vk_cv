#version 450
#extension GL_EXT_control_flow_attributes : enable
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data { uint[5][5] data; }
data;

void main() {
  uint idx = gl_GlobalInvocationID.x;
  data.data[idx] *= 12;
}