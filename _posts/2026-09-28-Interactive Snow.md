---
layout: post
title: Terrain Snow Deformation shader with tessellation and sliding window
date: 2026/06/19 09:02:00
description: infinite range interactive snow deformation shader without raycasting 
tags: shader
categories: devlog
thumbnail: assets/img/2_4.png
---

#### Snow Deformation
I always wondered snows in games were made, so I learned a lot of new materials and made my own version:

{% include video.liquid path="assets/video/snow.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=false%}

First, I followed [this cool tutorial](https://minionsart.github.io/tutorials/Posts.html?post=interactive_snow_urp) made by MinionsArt. And made a basic version of interactive snow. The basic idea is using a top-down camera and capture the trail made by particles and store the result into a render texture. Then in the snow shader, we read the texture by the vertex's horizontal world position as the uv, and push down the vertices that's inside a trail. Here's the result:

{% include figure.liquid loading="eager" path="assets/img/2_1.png" class="img-fluid rounded z-depth-1" %}

Currently the shader does not take height of the object into consideration, so when the player jumps, the trail will still be generated. Also, the exact shape of the object is not captured, so we would have to make a custom particle shape for every object if we want the snow trail to be accurate.

To add support for jumping, I changed the particle systems approach to custom render texture shader that accumulates the previous buffer with existing one. [Here](https://docs.unity3d.com/Packages/com.unity.shadergraph@17.0/manual/Custom-Render-Texture-Example.html)'s a example of Custom Render Texture Shader Graph. This way we don't have to rely on particle system to generate the trail.

But how do we support jumping or non-flat terrains?  We can capture the depth of the objects and compare them with height of the terrain. I changed the camera to be bottom-up so that it captures the objects' bottom parts depth correctly. Then we get the penetration amount by subtracting the object's depth value by the terrain's depth value. The Shader Graph:

{% include figure.liquid loading="eager" path="assets/img/2_2.png" class="img-fluid rounded z-depth-1" %}

Then I started thinking how can I improve the visuals even more. I read [Deferred Snow Deformation in Rise of the Tomb Raider](https://www.taylorfrancis.com/chapters/edit/10.1201/b21261-5/deferred-snow-deformation-rise-tomb-raider-anton-kai-michels-peter-sikachev) and they made an evevation around the snow depression to optimize the visuals:

{% include figure.liquid loading="eager" path="assets/img/2_3.png" class="img-fluid rounded z-depth-1" %}

So I did the same, to calculate the elevation around the snow trail, I used a compute shader to calculate the rim of the trail, then blurred the rim a little in another pass.

```GLSL
#pragma kernel ComputeRimCenter
[numthreads(8,8,1)]
void ComputeRimCenter (uint3 id : SV_DispatchThreadID)
{
    if (id.x >= size.x || id.y >= size.y)
    {
        return;
    }
    int2 coord = id.xy;
    float trail = Depth[coord].r;
    //there's no trail, detect if it's edge
    if (trail <= 0)
    {
        float2 l = Depth[coord+int2(-elevationRadius,0)].rg;
        float2 r = Depth[coord+int2(elevationRadius,0)].rg;
        float2 u = Depth[coord+int2(0,-elevationRadius)].rg;
        float2 d = Depth[coord+int2(0,elevationRadius)].rg;
        float2 l1 = Depth[coord+int2(-elevationRadius+2,0)].rg;
        float2 r1 = Depth[coord+int2(elevationRadius-2,0)].rg;
        float2 u1 = Depth[coord+int2(0,-elevationRadius+2)].rg;
        float2 d1 = Depth[coord+int2(0,elevationRadius-2)].rg;
        
        //it is edge, mark it and record the depth
        if (l.x>0&&l1.x<=0 || r.x>0&&r1.x<=0 || u.x>0&&u1.x<=0 || d.x>0&&d1.x<=0)
        {
            float maxLifeTime = max(l.g, max(r.g, max(u.g, d.g)));
            Rim[coord] = elevationStrength * maxLifeTime / lifeTime;
        }
        //not an edge
        else
        {
            Rim[coord] = 0;
        }
    }
    //it's trail
    else
    {
        Rim[coord] = 0;
    }
}

#pragma kernel BlurRim
[numthreads(8,8,1)]
void BlurRim (uint3 id : SV_DispatchThreadID)
{
    if (id.x >= size.x || id.y >= size.y)
        return;
    int2 coord = id.xy;
    float3x3 c = float3x3(
        0.0625,0.125,0.0625,
        0.125,0.25,0.125,
        0.0625,0.125,0.0625);
    float col = 0;
    for (int i = -1; i <= 1; ++i)
    {
        for (int j = -1; j <= 1; ++j)
        {
            col += Rim[coord+int2(i,j)] * c[i+1][j+1];
        }
    }
    BlurredRim[coord] = col;
}
```

With some adjustment to the shader graph, now we have elevation:

{% include figure.liquid loading="eager" path="assets/img/2_4.png" class="img-fluid rounded z-depth-1" %}

Currently, the snow shader used tessellation to tessellate nearby areas around the camera to give it more detail. This can be improved by only tessellating where snow is deformed. Modify the tessellation calculation code:

```GLSL
 float CalcDistanceTessFactor(float4vertex, float tess, float minTess)
 {
     float3 worldPos = mul(unity_ObjectToWorld, vertex).xyz;
     float2 uv = worldPos.xz/ (_OrthographicCamSize *2);
     uv += 0.5f;
     uv = frac(uv);
     
     float dist = distance(_TrailCameraPosition.xz, worldPos.xz);
     float Trail = 0;
     if (dist <= _OrthographicCamSize)
     {
         Trail = abs(GetTrail(uv));
     }
     return Trail > 0 ? tess : minTess;
 }
```
Then only the deformed area is tessellated:

{% include figure.liquid loading="eager" path="assets/img/2_5.png" class="img-fluid rounded z-depth-1" %}

I also spawned some snow particles around the trail with Graphics.RenderMeshIndirect. All spawning and positioning is calculated in the compute shader, and the particles are stored in an AppendStructuredBuffer:

```GLSL
#pragma kernel ComputeTrailParticles
[numthreads(8,8,1)]
void ComputeTrailParticles (uint3 id : SV_DispatchThreadID)
{
    if (id.x >= size.x || id.y >= size.y)
        return;
    float2 coord = id.xy;
    float pixelSize = 2*orthoCamSize/size;
    float2 uv = ((float2)coord / size) * float2(1, -1);
    float2 camPosPixelPerfect = round(cameraPosition/pixelSize)*pixelSize;
    float2 camPosDifference = (camPosPixelPerfect-cameraPosition)*float2(-1, 1);
    float2 posPixelPerfect = camPosPixelPerfect + (uv+float2(-0.5,0.5)) * orthoCamSize * 2;
    float2 slideWindowUV = (posPixelPerfect / (orthoCamSize*2));
    slideWindowUV -= float2(-0.5, 0.5);
    slideWindowUV = frac(slideWindowUV);
    int2 slideWindowCoord = (int2)(slideWindowUV * size);
    if (Rim[slideWindowCoord].r > 0 && random(posPixelPerfect) <= particleSpawnPossibility)
    {
        float scale = (random(posPixelPerfect+0.2)*0.3+0.2)*Rim[slideWindowCoord].r*3;
        if (scale > 0.2)
        {
            
            float2 terrainUv = coord / size;
            terrainUv -= camPosDifference / (2*orthoCamSize);
            float height = lerp(orthoCamFarPlane, 0, terrainDepth.SampleLevel(linearClampSampler, terrainUv, 0)+ Depth[slideWindowCoord].r*Depth[slideWindowCoord].g);
            height += Rim[slideWindowCoord].r*0.2 ;
            Particle particle;
            particle.scale = scale;
            particle.position = float3(posPixelPerfect.x, height, posPixelPerfect.y);
            particle.rotation = QuaternionFromYRotation(posPixelPerfect);
            _Particles.Append(particle);
        }
    }
}
```

The results look still a little weird, because there is only one particle mesh type.
{% include figure.liquid loading="eager" path="assets/img/2_6.png" class="img-fluid rounded z-depth-1" %}