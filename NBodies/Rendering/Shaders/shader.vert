#version 330 core

in vec4 aObjColor;
in vec3 aPosition;

out vec4 objColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform int isMesh;

void main(void)
{
    gl_Position = vec4(aPosition, 1.0) * model * view * projection;

	if (isMesh == 1) 
	{
		// Red color for mesh cubes...
		//objColor = vec4(1, 0, 0, 1);
		objColor = vec4(0.92, 0.58, 0.12, 0.8);
	}
	else 
	{
		objColor = aObjColor;
	}
}
