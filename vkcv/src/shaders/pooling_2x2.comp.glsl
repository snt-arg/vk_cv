#version 450

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, r8) uniform readonly image2D inputImage;
layout(set = 0, binding = 1, r8) uniform writeonly image2D resultImage;

// row major
layout(constant_id = 2) const int min_max = 1;

float op(in float a, in float b)
{
    if (min_max == 0) {
        return min(a, b);
    } else {
        return max(a, b);
    }
}

void main()
{
    ivec2 id = ivec2(gl_GlobalInvocationID.xy);
    ivec2 coord = id * 2;

    float res = imageLoad(inputImage, coord + ivec2(0, 0)).r;
    res = op(res, imageLoad(inputImage, coord + ivec2(1, 0)).r);
    res = op(res, imageLoad(inputImage, coord + ivec2(0, 1)).r);
    res = op(res, imageLoad(inputImage, coord + ivec2(1, 1)).r);

    imageStore(resultImage, id, vec4(res));
}