#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D texture0;

uniform vec2 offset;
uniform int horizontal;
uniform float weight[5] = float[](0.2270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162);
uniform int copy;

void main()
{
	if (copy == 0) { // Blur.
		vec4 c = vec4(0);
		vec2 uv = TexCoords;
		c += 5.0 * texture2D(texture0, uv - offset);
		c += 6.0 * texture2D(texture0, uv);
		c += 5.0 * texture2D(texture0, uv + offset);
		FragColor = c / 16.0;
	}
	else { // Just pass through.
		FragColor = texture2D(texture0, TexCoords);
	}
}