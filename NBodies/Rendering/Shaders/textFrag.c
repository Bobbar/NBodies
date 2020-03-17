#version 450 core
in vec2 vs_textureOffset;
in vec4 vs_color;
uniform sampler2D textureObject;
out vec4 color;

void main(void)
{
	vec4 alpha = texture(textureObject, vs_textureOffset);

	if (alpha.r < 0.1f)
		discard;
	
	color = vs_color;
	color.a = alpha.r;
}