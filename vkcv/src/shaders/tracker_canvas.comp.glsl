#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout(set = 0, binding = 0, r8) uniform readonly image2D inputImage;
layout(set = 0, binding = 1, r8) uniform writeonly image2D resultImage;

void main()
{
    ivec2 id = ivec2(gl_GlobalInvocationID.xy);

    float r = imageLoad(inputImage, id).r;
    imageStore(resultImage, id, vec4(r));
}