#version 450 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D scene;
uniform sampler2D bloomBlur;
uniform bool bloom;
uniform float exposure;
uniform float gamma;

void main()
{
	vec4 hdrColor = texture2D(scene, TexCoords);
	vec4 bloomColor = texture2D(bloomBlur, TexCoords);

	vec4 result = hdrColor;
	if (bloom)
	{
		bloomColor = pow(bloomColor, vec4(1.0 / gamma));
		bloomColor += vec4(1.0) - exp(-bloomColor * exposure);

		result += bloomColor;
	}

	FragColor = result;
}