#version 450 core

layout (location = 0) in vec3 cubeVert;
layout (location = 1) in vec3 aObjColor;
layout (location = 2) in vec4 aPosition;
layout (location = 3) in vec3 cubeNormal;

out vec3 objectColor;
out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float nearPlaneHeight;
uniform int usePoint;

void main(void)
{
	if (usePoint == 1 || usePoint == 2) 
	{
		objectColor = aObjColor;
		vec4 position = vec4(aPosition.xyz, 1.0) * model * view;
		float dist = -position.z;
		//float dist = distance(aPosition.xyz, position.xyz);
		gl_Position = position *  projection;
		FragPos = gl_Position.xyz;
		gl_PointSize = (nearPlaneHeight * (aPosition.w * 2.0f)) / gl_Position.w;
	}
	else
	{
		gl_Position = vec4((cubeVert * aPosition.w) + aPosition.xyz, 1.0) * model * view * projection;
		FragPos = vec3(vec4((cubeVert * aPosition.w) + aPosition.xyz, 1.0) * model);
		Normal = cubeNormal * mat3(transpose(inverse(model)));
		objectColor = aObjColor;
	}
}
