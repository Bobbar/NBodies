#version 450 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D blurTex;
//uniform vec2 offset;
uniform vec4 offset;
uniform int horizontal;
uniform int copy;

vec4 GaussianBlur( sampler2D tex0, vec2 centreUV, vec2 halfPixelOffset, vec2 pixelOffset );

void main()
{
	if (copy == 0) { // Blur.

		// vec4 c = vec4(0);
		// vec2 uv = TexCoords;
		// c += 5.0 * texture2D(blurTex, uv - offset);
		// c += 6.0 * texture2D(blurTex, uv);
		// c += 5.0 * texture2D(blurTex, uv + offset);
		// FragColor = c / 16.0;



	if (horizontal == 1){
		// FragColor.xyz = GaussianBlur( blurTex, TexCoords, vec2( offset.z, 0 ), vec2( offset.x, 0 ) );
		// FragColor.w = 1.0;//0.05;

		FragColor = GaussianBlur( blurTex, TexCoords, vec2( offset.z, 0 ), vec2( offset.x, 0 ) );
		
	}
	else {
		// FragColor.xyz = GaussianBlur( blurTex, TexCoords, vec2( 0, offset.w ), vec2( 0, offset.y ) );
    	// FragColor.w = 1.0;//0.05;

		FragColor = GaussianBlur( blurTex, TexCoords, vec2( 0, offset.w ), vec2( 0, offset.y ) );
    	
	}

	


	}
	else { // Just pass through.
		FragColor = texture2D(blurTex, TexCoords);
	}
}



vec4 GaussianBlur( sampler2D tex0, vec2 centreUV, vec2 halfPixelOffset, vec2 pixelOffset )                                                                           
{                                                                                                                                                                    
    vec4 colOut = vec4(0);                                                                                                                                   
                                                                                                                                                                     
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////;
    // Kernel width 35 x 35
    //
    const int stepCount = 9;
    
    const float gWeights[stepCount] ={
       0.10855,
       0.13135,
       0.10406,
       0.07216,
       0.04380,
       0.02328,
       0.01083,
       0.00441,
       0.00157
    };
    const float gOffsets[stepCount] ={
       0.66293,
       2.47904,
       4.46232,
       6.44568,
       8.42917,
       10.41281,
       12.39664,
       14.38070,
       16.36501
    };

//  const int stepCount = 4;
//     //
//     const float gWeights[stepCount] ={
//       0.2496147,
// 0.1924633,
// 0.05147626,
// 0.006445717
//     };
//     const float gOffsets[stepCount] ={
//       0.6443417,
// 2.378848,
// 4.291111,
// 6.216607
//     };

// const int stepCount = 5;
//     //
//     const float gWeights[stepCount] ={
//       0.1995468,
// 0.1894521,
// 0.08376212,
// 0.02321143,
// 0.004027504
//     };

//     const float gOffsets[stepCount] ={
//     0.6531861,
// 2.425468,
// 4.368035,
// 6.314115,
// 8.2647867
//     };

// const int stepCount = 6;
//     //
//     const float gWeights[stepCount] ={
//      0.165014,
// 0.1750711,
// 0.1011206,
// 0.04267556,
// 0.01315655,
// 0.002962173
//     };

//     const float gOffsets[stepCount] ={
//   0.6577193,
// 2.450166,
// 4.41096,
// 6.372852,
// 8.336262,
// 10.30153
//     };

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////;
                                                                                                                                                                     
    for( int i = 0; i < stepCount; i++ )                                                                                                                             
    {                                                                                                                                                                
        vec2 texCoordOffset = gOffsets[i] * pixelOffset;                                                                                                           
        vec4 col = texture( tex0, centreUV + texCoordOffset ) + texture( tex0, centreUV - texCoordOffset );                                                
        colOut += gWeights[i] * col;                                                                                                                               
    }                                                                                                                                                                
                                                                                                                                                                     
    return colOut;                                                                                                                                                   
}        