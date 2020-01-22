#version 330 core

in vec4 aObjColor;
in vec3 aPosition;

out vec4 objColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 color;
uniform float alpha;


void main(void)
{
    gl_Position = vec4(aPosition, 1.0) * model * view * projection;
	objColor = vec4(color, alpha);
	
}
