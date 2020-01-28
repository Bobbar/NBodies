#version 330 core

layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec3 aObjColor;
layout (location = 2) in vec4 aOffset;
layout (location = 3) in vec3 aNormal;

out vec3 objectColor;
out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
//uniform float alpha;

void main(void)
{
	gl_Position = vec4((aPosition * aOffset.w) + aOffset.xyz, 1.0) * model * view * projection;

	FragPos = vec3(vec4((aPosition * aOffset.w) + aOffset.xyz, 1.0) * model);
	//FragPos = vec3(vec4(aPosition, 1.0) * model);

	Normal = aNormal * mat3(transpose(inverse(model)));
	//Normal = aNormal * aOffset.w * mat3(transpose(inverse(model)));

   // objectColor = vec4(aObjColor, alpha);
    objectColor = aObjColor;

}
