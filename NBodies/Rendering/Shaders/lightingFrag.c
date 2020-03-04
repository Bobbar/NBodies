﻿#version 330 core
out vec4 FragColor;

//In order to calculate some basic lighting we need a few things per model basis, and a few things per fragment basis:
//uniform vec3 objectColor; //The color of the object.
uniform vec3 lightColor; //The color of the light.
uniform vec3 lightPos; //The position of the light.
uniform vec3 viewPos; //The position of the view and/or of the player.
uniform float alpha;
uniform int noLight;
uniform sampler2D texture0;
uniform int usePoint;

float Ns = 25;//250;
//vec4 mat_specular = vec4(1);
vec4 mat_specular = vec4(0.4);

//vec4 light_specular = vec4(1);
vec4 light_specular = vec4(0.5);

in vec3 Normal; //The normal of the fragment is calculated in the vertex shader.
in vec3 FragPos; //The fragment position.
in vec3 objectColor;

void main()
{
	if (usePoint == 1)
	{
		vec4 bodyColor = vec4(objectColor, alpha);
		vec4 tex = texture2D(texture0, vec2(gl_PointCoord.x, gl_PointCoord.y));
		FragColor = bodyColor * tex;
	}
	else if (usePoint == 2)
	{
		vec4 bodyColor = vec4(objectColor, alpha);
		//vec3 lightDir = vec3(0.577, 0.577, 0.577);
		vec3 lightDir = normalize(FragPos - normalize(lightPos));

		vec3 N;
		//N.xy = gl_PointCoord.xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
		N.xy = gl_PointCoord.xy*vec2(-2.0, 2.0) + vec2(1.0, -1.0);

		float mag = dot(N.xy, N.xy);
		if (mag > 1) discard;   // kill pixels outside circle
		N.z = sqrt(1 - mag);

		// calculate lighting
		float diffuse = max(0.0, dot(lightDir, N));
		//FragColor = bodyColor * diffuse;

		//vec3 eye = vec3(0.0, 0.0, 1.0);
		vec3 eye = vec3(0.0, 0.0, 0.0);
		vec3 halfVector = normalize(eye + lightDir);
		float spec = max(pow(dot(N, halfVector), Ns), 0.);
		vec4 S = light_specular*mat_specular* spec;
		FragColor = bodyColor * diffuse + S;

	}
	else
	{
		if (noLight == 1)
		{
			float dist = distance(viewPos, FragPos);
			dist += 0.001f;
			float cutoff = 50.0f;//100.0f;
			float red = clamp(cutoff / dist, 0.1f, 1);
			float black = clamp(dist / cutoff, 0, 0.1f);
			FragColor = vec4(red, black, black, 0.8f);

			//FragColor = vec4(objectColor, alpha);
		}
		else
		{
			//The ambient color is the color where the light does not directly hit the object.
			//You can think of it as an underlying tone throughout the object. Or the light coming from the scene/the sky (not the sun).
			float ambientStrength = 0.1;
			vec3 ambient = ambientStrength * lightColor;

			//We calculate the light direction, and make sure the normal is normalized.
			vec3 norm = normalize(Normal);
			vec3 lightDir = normalize(lightPos - FragPos); //Note: The light is pointing from the light to the fragment

														   //The diffuse part of the phong model.
														   //This is the part of the light that gives the most, it is the color of the object where it is hit by light.
			float diff = max(dot(norm, lightDir), 0.0); //We make sure the value is non negative with the max function.
			vec3 diffuse = diff * lightColor;


			//The specular light is the light that shines from the object, like light hitting metal.
			//The calculations are explained much more detailed in the web version of the tutorials.
			float specularStrength = 0.8; //0.5;
			vec3 viewDir = normalize(viewPos - FragPos);
			vec3 reflectDir = reflect(-lightDir, norm);
			float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32); //The 32 is the shininess of the material.
			vec3 specular = specularStrength * spec * lightColor;

			//At last we add all the light components together and multiply with the color of the object. Then we set the color
			//and makes sure the alpha value is 1
			vec3 result = (ambient + diffuse + specular) * objectColor;
			FragColor = vec4(result, alpha);

			//Note we still use the light color * object color from the last tutorial.
			//This time the light values are in the phong model (ambient, diffuse and specular)
		}

	}

}