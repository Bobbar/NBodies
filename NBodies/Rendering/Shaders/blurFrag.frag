#version 450 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D blurTex;
uniform vec2 offset;
uniform int horizontal;
uniform int copy;

void main()
{
	if (copy == 0) { // Blur.
		vec4 c = vec4(0);
		vec2 uv = TexCoords;
		c += 5.0 * texture2D(blurTex, uv - offset);
		c += 6.0 * texture2D(blurTex, uv);
		c += 5.0 * texture2D(blurTex, uv + offset);
		FragColor = c / 16.0;
	}
	else { // Just pass through.
		FragColor = texture2D(blurTex, TexCoords);
	}
}