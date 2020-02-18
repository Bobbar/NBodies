#version 400

in vec4 color;
//in float massColor;
uniform sampler2D texture0;
uniform float alpha;


out vec4 frag_colour;

void main() {
	//vec4 starColor = mix(color, vec4(1, 1, 1, 1), clamp(massColor, 0, 1));
	//vec4 starColor = color;//mix(color, vec4(1, 1, 1, 1));
	//starColor = vec4(starColor.xyz, 10.0f);
	//frag_colour = starColor;
	/*vec4 oCol = starColor * texture(cloud, vec2(gl_PointCoord.x, gl_PointCoord.y));
	oCol = vec4(oCol.xyz, (oCol.w + 0.3f));
	frag_colour = cCol;*/

	//vec4 tex = texture2D(cloud, vec2(gl_PointCoord.x, gl_PointCoord.y));

	//vec4 tex = texture2D(cloud, vec2(gl_PointCoord.x * 0.0156f, gl_PointCoord.y * 0.0156f));
	
	// Can't seem to get the texture to bind properly...
	//vec4 tex = texture2D(texture0, vec2(0.5f, 0.5f));
	
	
	vec4 starColor = vec4(color.xyz, alpha);
	vec4 tex = texture2D(texture0, vec2(gl_PointCoord.x, gl_PointCoord.y));
	frag_colour = starColor * tex;




	/*if (tex.r == 0 || tex.g == 0)
		frag_colour = vec4(1, 1, 1, 1);
	else
		frag_colour = starColor * tex;*/




	//frag_colour = starColor * gl_PointCoord.y; //displays white gradient

	//frag_colour = starColor * tex;

	/*frag_colour.r = 1.0f;

	frag_colour.a = tex.r;*/



	//frag_colour = starColor * texture(cloud, vec2(gl_PointCoord.x, gl_PointCoord.y));


	//frag_colour = vec4(color.xyz, alpha);
};