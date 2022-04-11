#version 450

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba32f) uniform readonly image2D inputImage;
layout(set = 0, binding = 1, rgba32f) uniform writeonly image2D resultImage;

void main()
{
    ivec2 id = ivec2(gl_GlobalInvocationID.xy);

    // Note: The RPi misbehaves when using the sampler method
    //       However, this approach works.
    //       Furthermore, it has no performance penalty on the RPi.
    ivec2 p = id * 2;
    vec4 d = imageLoad(inputImage, p + ivec2(0, 0));
    d += imageLoad(inputImage, p + ivec2(0, 1));
    d += imageLoad(inputImage, p + ivec2(1, 0));
    d += imageLoad(inputImage, p + ivec2(1, 1));

    imageStore(resultImage, id, d * 0.25);
}
