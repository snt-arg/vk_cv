#version 450

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, r8) uniform readonly image2D inputImage;
layout(set = 0, binding = 1, r8) uniform writeonly image2D resultImage;

layout(constant_id = 2) const float m11 = -1.0;
layout(constant_id = 3) const float m12 = 0.0;
layout(constant_id = 4) const float m13 = 0.0;
layout(constant_id = 5) const float m21 = 0.0;
layout(constant_id = 6) const float m22 = -1.0;
layout(constant_id = 7) const float m23 = 0.0;
layout(constant_id = 8) const float m31 = 0.0;
layout(constant_id = 9) const float m32 = 0.0;
layout(constant_id = 10) const float m33 = 2.0;
layout(constant_id = 11) const float offset = 0.5;
layout(constant_id = 12) const float denom = 2.0;
// layout(constant_id = 1) const uint THREADS_PER_GROUP_Y = 16;
// layout(constant_id = 2) const uint THREADS_PER_GROUP = 64;

// shared float sharedData[THREADS_PER_GROUP_X][THREADS_PER_GROUP_Y];

// 128 bytes max.
// layout(push_constant) uniform PushConstants {
//   float[9] kernel;
//   float offset;
//   float denom;
// }
// pc;

// soebel kernel can be seperated
// [1 2 1].T x [-1 0 1]

// const float[9] kernel = float[](m11, m12, m13, m21, m22, m23, m31, m32, m33);
// const float denom = 2.0;
// const float offset = 0.5;

float conv(in float[9] data)
{
    float acc = m11 * data[0] + m12 * data[1] + m13 * data[2] + //
        m21 * data[3] + m22 * data[4] + m23 * data[5] + //
        m31 * data[6] + m32 * data[7] + m33 * data[8];

    return clamp(acc / denom + offset, 0.0, 1.0);
}

void main()
{
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

    // sharedData[lid.x][lid.y] = imageLoad(inputImage, ivec2(id.x, id.y)).r;
    // memoryBarrierShared();
    // barrier();

    float avg[9];
    // avg[0] = sharedData[lid.x - 1][lid.y - 1];
    // avg[1] = sharedData[lid.x - 1][lid.y + 0];
    // avg[2] = sharedData[lid.x - 1][lid.y + 1];
    // avg[3] = sharedData[lid.x + 0][lid.y - 1];
    // avg[4] = sharedData[lid.x + 0][lid.y + 0];
    // avg[5] = sharedData[lid.x + 0][lid.y + 1];
    // avg[6] = sharedData[lid.x + 1][lid.y - 1];
    // avg[7] = sharedData[lid.x + 1][lid.y + 0];
    // avg[8] = sharedData[lid.x + 1][lid.y + 1];

    avg[0] = imageLoad(inputImage, ivec2(id.x - 1, id.y - 1)).r;
    avg[1] = imageLoad(inputImage, ivec2(id.x - 1, id.y - 0)).r;
    avg[2] = imageLoad(inputImage, ivec2(id.x - 1, id.y + 1)).r;
    avg[3] = imageLoad(inputImage, ivec2(id.x - 0, id.y - 1)).r;
    avg[4] = imageLoad(inputImage, ivec2(id.x - 0, id.y - 0)).r;
    avg[5] = imageLoad(inputImage, ivec2(id.x - 0, id.y + 1)).r;
    avg[6] = imageLoad(inputImage, ivec2(id.x + 1, id.y - 1)).r;
    avg[7] = imageLoad(inputImage, ivec2(id.x + 1, id.y - 0)).r;
    avg[8] = imageLoad(inputImage, ivec2(id.x + 1, id.y + 1)).r;

    // int n = 0;
    // for (int i = -1; i < 2; ++i) {
    //   for (int j = -1; j < 2; ++j) {
    //     avg[n] = sharedData[lid.x + i][lid.y + j];
    //     n++;
    //   }
    // }

    vec4 res = vec4(conv(avg));

    imageStore(resultImage, ivec2(id.xy), res);
}