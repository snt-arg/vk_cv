#version 450

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, r8) uniform readonly image2D inputImage;
layout(set = 0, binding = 1, r8) uniform image2D resultImage;

layout(constant_id = 2) const float m1 = 1.0;
layout(constant_id = 3) const float m2 = 2.0;
layout(constant_id = 4) const float m3 = 1.0;

layout(constant_id = 5) const float m4 = 1.0;
layout(constant_id = 6) const float m5 = 0.0;
layout(constant_id = 7) const float m6 = -1.0;

layout(constant_id = 8) const float offset = 0.0;
layout(constant_id = 9) const float denom = 1.0;

layout(constant_id = 10) const int v_pass = 0;

float conv(in float[3] kernel, in float[3] data)
{
    float acc;
    if (v_pass == 0) {
        acc = m1 * data[0] + m2 * data[1] + m3 * data[2];
    } else {
        acc = m4 * data[0] + m5 * data[1] + m6 * data[2];
    }

    return clamp(acc / denom + offset, 0.0, 1.0);
}

void main()
{
    // horizontal and vertical pass
    // the current pass is indicated by v_pass
    uvec2 id = gl_GlobalInvocationID.xy;

    float avg[3];
    avg[0] = imageLoad(inputImage, ivec2(id.x - 1, id.y)).r;
    avg[1] = imageLoad(inputImage, ivec2(id.x + 0, id.y)).r;
    avg[2] = imageLoad(inputImage, ivec2(id.x + 1, id.y)).r;

    vec4 res;
    if (v_pass == 0) {
        res = vec4(conv(float[](m1, m2, m3), avg));
    } else {
        res = vec4(conv(float[](m4, m5, m6), avg));
    }

    imageStore(resultImage, ivec2(id.yx), res);
}