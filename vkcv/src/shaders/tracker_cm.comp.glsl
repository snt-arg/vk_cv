#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout(set = 0, binding = 0, r8) uniform readonly image2D inputImage;
layout(set = 0, binding = 1, rgba16f) uniform writeonly image2D resultImage;

layout(constant_id = 3) const float inv_width = 1.0;
layout(constant_id = 4) const float inv_height = 1.0;

void main()
{
    ivec2 id = ivec2(gl_GlobalInvocationID.xy);
    const vec2 inv_size = vec2(inv_width, inv_height);

    // coordinate mask
    float r = imageLoad(inputImage, id).r;
    vec2 d = r * vec2(id) * inv_size;

    imageStore(resultImage, id, vec4(d, r, r));
}