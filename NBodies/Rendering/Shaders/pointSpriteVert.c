#version 400

layout(location = 0) in vec3 aObjColor;
layout(location = 1) in vec4 aPosition;

//in vec4 vp;
//in vec4 cp;
//in float mp;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
//uniform vec3 viewPos;

out vec4 color;


out gl_PerVertex{
	vec4 gl_Position;
float gl_PointSize;
float gl_ClipDistance[];

};

void main() {
	color = vec4(aObjColor, 1.0);

	vec4 position = vec4(aPosition.xyz, 1.0) * model * view;
	//vec4 position = vec4((aPosition.xyz * aPosition.w), 1.0) * model * view;

	//float dist = -position.z;
	float dist = -position.z;//distance(aPosition.xyz, viewPos);

	gl_Position = position *  projection;
	//gl_Position = vec4(aPosition) * view * projection;
	//gl_PointSize = 5.0f;//aPosition.w;
	gl_PointSize = 1000.0f * aPosition.w / dist;

	
};