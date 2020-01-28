#version 330 core

layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec3 aObjColor;
layout (location = 2) in vec4 aOffset;

out vec4 objColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float alpha;

void main(void)
{
	gl_Position = vec4((aPosition * aOffset.w) + aOffset.xyz, 1.0) * model * view * projection;
    objColor = vec4(aObjColor, alpha);
}
