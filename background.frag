#version 450 // GLSL version 4.5

layout(push_constant) uniform Push {
    float time;
};
layout(location = 0) in vec2 position;
layout(location = 0) out vec4 outColor;

// Adapted from https://github.com/blender/blender/blob/main/intern/cycles/kernel/svm/voronoi.h
// and https://www.shadertoy.com/view/flSGDK

// Source - https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
// Answer posted by appas, modified by community. See post 'Timeline' for change history
// Retrieved 2026-01-13, License - CC BY-SA 4.0
float hash2(vec2 x){
    return fract(sin(dot(x, vec2(12.9898, 78.233))) * 43758.5453);
}
float hash(float x) { return fract(x + 1.3215 * 1.8152); }

vec2 rehash2(float x) { return vec2(hash(((x + 0.5283) * 59.3829) * 274.3487), hash(((x + 0.8192) * 83.6621) * 345.3871)); }

vec2 voronoi(vec2 coord, float scale) {
    vec2 pos = coord * scale;
    ivec2 cellPos = ivec2(floor(pos));
    vec2 localPos = pos - vec2(cellPos);

    float minDist = 9999.0;
    ivec2 targetOffset = ivec2(0);
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            ivec2 cellOffset = ivec2(x, y);
            float cellHash = hash2(vec2(cellPos + cellOffset));
            vec2 pointPos = cellOffset + rehash2(cellHash);
            float dist = length(localPos - pointPos);
            if (dist < minDist) {
                minDist = dist;
                targetOffset = cellOffset;
            }
        }
    }

    ivec2 targetCell = cellPos + targetOffset;
    float targetCellHash = hash2(vec2(targetCell));
    return vec2(minDist, targetCellHash);
}

void main() {
    vec2 v1_out = voronoi(gl_FragCoord.xy + vec2(200, 100) * sin(time/60.0 * 6.283185), 0.01);
    vec2 v2_out = voronoi(gl_FragCoord.xy + vec2(-500, 700) * sin(time/60.0 * 6.283185 + 2), 0.005);
    float x = v1_out.x * v1_out.x * 0.5 + v2_out.x * v2_out.x;
    outColor = mix(vec4(0.0, 0.078, 0.25, 1.0), vec4(0.02, 0.762, 0.863, 1.0), x);
}