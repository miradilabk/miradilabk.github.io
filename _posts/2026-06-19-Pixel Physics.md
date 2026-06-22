---
layout: post
title: Simulating fluid, smoke, and sand simutaneously on the GPU
date: 2026/06/19 09:02:00
description: 2D pixel physics simulation
tags: physics, 2D
categories: devlog
---

#### Simple Noita-like Demo
I really enjoyed playing [Noita](https://store.steampowered.com/app/881100/Noita/) and was really impressed by the 2d pixel physics simulation in the game. I watched the [GDC talk](https://www.youtube.com/watch?v=prXuyMCgbTc) about the tech behind Noita and made a small demo in Unity replicating the pixel physics part.

{% include video.liquid path="assets/video/pixel_physics_demo.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay: false%}

This demo runs on CPU and it's pretty simple. In fixed update we loop over every pixel and check which type of pixel it is and try to move it in every direction it can. For example, sand pixel can fall directly downwards(0,-1), or bottom left(-1, -1), bottom right(1, -1). Water can also move horizontally (-1,0), (1,0). Gas can move up instead so it's the reversed version of water.

Part of the code in FixedUpdate
```csharp
for (int j = height-1; j >= 0; j--)
{
    int dir = Random.Range(0, 2);
    for (int i = dir==0?0:width-1; dir== 0?i < width : i >= 0;)
    {
        if (_buffer[i,j].isParticle)
        {
            if (dir == 0) i++;
            else i--;
            continue;
        }
        short elementId = _buffer[i,j].elementId;
        var element = elements[elementId];
        if (elementId != 0 && elementId < elements.Length)
        {
            for (int k = 0; k < element.directions.Length; k++)
            {
                Vector2Int direction = element.directions[k];
                int x = i + direction.x;
                int y = j + direction.y;
                if (!IsInBounds(x, y)) continue;
                var neighborElement = elements[_buffer[x, y].elementId];
                short res1=0, res2=0;
                var hasReaction =
                PixelChemistryController.instance.CheckReaction(elementId, _buffer[x, y].elementId, 
                    ref res1, ref res2);
                
                if (hasReaction)
                {
                    ChangePixel(i,j,res1);
                    ChangePixel(x,y,res2);
                    break;
                }
                
                if (neighborElement.type < elements[elementId].type && Random.value*100 < element.movePossibility)
                {
                    SwapPixel(i, j, x, y);
                    break;
                }

                if (Random.value < 0.1f && neighborElement.type == ElementType.Liquid &&
                    elements[elementId].type == ElementType.Liquid && Random.value*100 < element.movePossibility)
                {
                    SwapPixel(i, j, x, y);
                    break;
                }
            }
        }

        if (dir == 0) i++;
        else i--;
    }
}
```

#### Fluid Simulation
While this is cool, I couldn't help but notice the pixel movement isn't very realistic. Especially for water and gas, it's not flowing or diffusing. So I started digging around to see if there's a way to simulate fluid more accurately. And indeed there is! I found [this tutorial](https://www.youtube.com/watch?v=rSKMYc1CQHE) is really good for beginners like me and I followed it step by step and made this: 

{% include video.liquid path="assets/video/fluid_simulation.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay: false%}

This fluid simulation runs on the GPU with compute shader. The main logic is no water particle cannot be too close to each other, but also has an attraction force to keep the water shape. To achieve that we need to calculate the density of the water particles, then we can calculate the pressure and the force to determine which direction the particle should go.

If we do this by brute force, it will be very slow calculating particle's density by looping over all the particles. Instead we can only check surrounding particles by partitioning the space into grids. Then we can only check the particles in the neighboring grids. We need spatial hashing to assign a grid index by particle position. Then we have to sort the particle indices by hashed grid index so that we can loop over all particles in any grid. Because we're on the GPU so we have to use the [Bitonic Merge Sort](https://quinstonpimenta.medium.com/bitonic-merge-sort-explanation-and-code-tutorial-5688bd3507fb) which is optimized for parallel processing.

```GLSL
float2 CalculateDensity(float2 pos)
{
    float density = 0;
    float nearDensity = 0;
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
        {
            float x = pos.x + smoothRadius * i;
            float y = pos.y + smoothRadius * j;
            uint thisHash = GetHashByPos(x,y);
            for (uint k = startIndices[thisHash]; k < particleCnt && spacialHash[spacialHashKeys[k]].x == thisHash; k++)
            {
                uint ii = spacialHash[spacialHashKeys[k]].y;
                if (isLiquidHidden[ii]) continue;
                float2 pos2 = predictions[ii];
                float dst = distance(pos, pos2);
                float w = SmoothKernel(dst);
                density += w;
                nearDensity += SmoothKernelSpiky(dst);
            }
        }
    return float2(density, nearDensity);
}
```

Now I have a pretty good water simulation. But I have to integrate this into the previous pixel physics demo. There are a couple of problems I have to solve:
1. fluid simulation is position-based particles, but I want a pixel-based water simulation.
2. How to make water particles collide with sand or other solid pixels? The previous demo runs on the CPU, which is pretty slow and not easy to make it interact with compute shader based fluid simulation.

<hr>
For the first problem, I added an extra kernel to the compute shader that can sample the density level for each pixel, then sample the density field into a texture, if the density for current pixel is higher than a certain threshold, this pixel is water pixel.

{% include video.liquid path="assets/video/pixel_water.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay: false%}

You may noticed that the water has a foam effect applied, that's because I calculated a velocity-based foam factor in the compute shader as well. Basically for each pixel, I find the nearest particle to this pixel, then calculate how different the velocity is compared to other nearby particles. The bigger the velocity difference, the stronger the foam.

Here's the compute shader kernel for density sampling and foam calculation.
```GLSL
#pragma kernel SampleDensityMap
[numthreads(8,8,1)]
void SampleDensityMap (uint2 id: SV_DispatchThreadID)
{
    float2 pos = (float2)id*boundaryParticleDist;
    if (pos.x > width || pos.y > height) return;
    float2 uv = pos/float2(width,height);
    int2 pixel = uv*boundaryTextureSize;
    
    
    float density = 0;
    uint nearestParticle = 0;
    float nearestDist = 100000;
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
        {
            float x = pos.x + smoothRadius * i;
            float y = pos.y + smoothRadius * j;
            uint thisHash = GetHashByPos(x,y);
            for (uint k = startIndices[thisHash]; k < particleCnt && spacialHash[spacialHashKeys[k]].x == thisHash; k++)
            {
                uint ii = spacialHash[spacialHashKeys[k]].y;
                if (isLiquidHidden[ii]) continue;
                float2 pos2 = predictions[ii];
                float dst = distance(pos, pos2);
                if (dst < nearestDist)
                {
                    nearestDist = dst;
                    nearestParticle = ii;
                }
                float w = SmoothKernel(dst);
                density += w;
            }
        }
    float densityRatio = density/targetDensity;
    if (densityRatio >= 0.7)
    {
        float2 nearestPos = predictions[nearestParticle];
        float2 velDif = 0;
        //velocity diff
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
            {
                float x = pos.x + smoothRadius * i;
                float y = pos.y + smoothRadius * j;
                uint thisHash = GetHashByPos(x,y);
                for (uint k = startIndices[thisHash]; k < particleCnt && spacialHash[spacialHashKeys[k]].x == thisHash; k++)
                {
                    uint ii = spacialHash[spacialHashKeys[k]].y;
                    if (isLiquidHidden[ii]) continue;
                    float2 pos2 = predictions[ii];
                    float dst = distance(nearestPos, pos2);
                    float w = SmoothKernel(dst);
                    if (isnan(velocities[ii].x)||isnan(velocities[ii].y)) continue;
                    velDif += abs(velocities[nearestParticle]-velocities[ii])*w;
                }
            }
        float foamRatio = min(1,(velDif.x+velDif.y)/foamVelocity);
        foamRatio *= foamRatio;
        fluidDensityMap[pixel]=densityRatio < 0.7 ? 0 : lerp(float4(0.05,0.53,0.8,.1), float4(1,1,1,1), foamRatio);
    }
    else
    {
        fluidDensityMap[pixel] = float4(0,0,0,0);
    }
}
```

<hr>
For the second problem, I converted the pixel physics script to run on compute shader. It was pretty straight forward, the kernel code:

```GLSL
#pragma kernel SimulateStep
[numthreads(8,8,1)]
void SimulateStep (uint2 id : SV_DispatchThreadID)
{
    /*if (id.x == 0 || id.x >= size.x-1 || id.y == 0 ||id.y == size.y-1)
    {
        Pixels[id] = float4(1,1,1,-1);
    }*/
    if (distance(id, drawingPixelPos) <= 5)
    {
        Pixels[id] = float4(pixelToDraw.rgb*(1-rand_1_05((float2)id/float2(320,180))*0.5),pixelToDraw.a);
    }
    if (id.y == 0 || Pixels[id].a == 0) return;
    bool isSand = Pixels[id].a > 0;
    if (isSand)
    {
        if (Pixels[id+int2(0,-1)].a == 0)
        {
            Pixels[id+int2(0,-1)] = Pixels[id];
            Pixels[id] = float4(0,0,0,0);
        }
        else
        {
            if (rand_1_05(id/float2(320,180)) <= 0.5)
            {
                if (id.x >0 && Pixels[id+int2(-1,-1)].a == 0)
                {
                    Pixels[id+int2(-1,-1)] = Pixels[id];
                    Pixels[id] = float4(0,0,0,0);
                }
                else if (id.x < 319 && Pixels[id+int2(1,-1)].a == 0)
                {
                    Pixels[id+int2(1,-1)] = Pixels[id];
                    Pixels[id] = float4(0,0,0,0);
                }
            }
            else
            {
                if (id.x < 319 && Pixels[id+int2(1,-1)].a == 0)
                {
                    Pixels[id+int2(1,-1)] = Pixels[id];
                    Pixels[id] = float4(0,0,0,0);
                }
                else if (id.x > 0 && Pixels[id+int2(-1,-1)].a == 0)
                {
                    Pixels[id+int2(-1,-1)] = Pixels[id];
                    Pixels[id] = float4(0,0,0,0);
                }
            }
        }
    }
}
```
Currently it has two types of pixel: powder and solid, determined by the alpha value of the pixel.

Next step is to tell the fluid simulation the boundary map. This map should contain the position and the normal vector of the boundary so that when a water particle hits a boundary, it can bounce to the right direction according to the normal vector. The map should also be updated in realtime so when player interacts with it the map will change.

For this I made a SDF(signed distance field) generator that runs on the GPU to generate boundary map based on the pixel physics map. This map contains the shortest distance to the nearest boundary surface. So on the boundary surface pixel, the value is 0, negative inside the boundary and positive outside the boundary. With this distance information, we can calculate the normal of the boundary at any given position.

```GLSL
float2 SDFGradient(float2 uv)
{
    const float eps = 0.003125;
    float dx = boundaryTexture.SampleLevel(sampler_LinearClamp, float2(uv.x+eps,uv.y),0).r -
        boundaryTexture.SampleLevel(sampler_LinearClamp, float2(uv.x-eps,uv.y),0).r;
    float dy = boundaryTexture.SampleLevel(sampler_LinearClamp, float2(uv.x,uv.y+eps),0).r -
        boundaryTexture.SampleLevel(sampler_LinearClamp, float2(uv.x,uv.y-eps),0).r;
    return normalize(float2(dx,dy));
}
```

The above code showes how to get the normal vector by signed distance field, it samples the SDF texture 4 times on four neighbor pixels, then calculate the distance difference and that gives us the normal vector. Then we can use this normal vector to resolve the collision between water particle and the boundary.

```GLSL
float2 size = float2(width, height);
float2 uv = pos/size;
//in a wall
float dist = boundaryTexture.SampleLeve(sampler_LinearClamp, uv, 0).r - 0.003125;
if (dist < 0)
{
    float2 normal = SDFGradient(uv);
    pos -= normal * dist*size;
    
    positions[i] -= normal * dist*size;
    float2 vn = dot(velocities[i], normal) * normal;
    float2 vt = velocities[i] - vn;
    velocities[i] -= vn * (1.1f);
    velocities[i] -= vt * 0.2f;
    
}
```

I used [Jump Flooding Algorithm](https://en.wikipedia.org/wiki/Jump_flooding_algorithm) to generate the SDF.

{% include figure.liquid loading="eager" path="assets/img/SDF.png" class="img-fluid rounded z-depth-1" %}

```GLSL
#pragma kernel JFAPass
[numthreads(8,8,1)]
void JFAPass(uint3 id : SV_DispatchThreadID) {
    float2 best = Result[id.xy];
    float2 pos = (float2)id.xy/size;
    float bestDist = (best.x < 0) ? 1e20 : distance(best, pos);

    for(int ox=-1; ox<=1; ox++) {
        for(int oy=-1; oy<=1; oy++) {
            int2 n = id.xy + int2(ox, oy) * stepSize;
            if (n.x < 0 || n.y < 0 || n.x >= size.x || n.y >= size.y) continue;

            float2 candidate = Result[n];
            if (candidate.x < 0) continue;

            float d = distance(candidate, pos);
            if(d < bestDist) {
                bestDist = d;
                best = candidate;
            }
        }
    }
    Result[id.xy] = best;
}
```

Now we have a fluid&solid&powder simulation!

{% include video.liquid path="assets/video/collision.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay: false%}

#### Smoke Simulation
Now that we have water and powder, there's one thing still missing: Smoke. After reading a bunch of papers that I couldn't fully understand and some really good tutorials. I found that smoke simulation is a type of fluid simulation, but using a very different technique than water simulation.

To simulate smoke, I found the most common approach is using Eulerian Fluid Simulation which is taught in [this great video](https://www.youtube.com/watch?v=iKAVRgIrUOU) and [this video](https://www.youtube.com/watch?v=Q78wvrQ9xsU&t=1856s), and [this blog](https://shahriyarshahrabi.medium.com/gentle-introduction-to-fluid-simulation-for-programmers-and-technical-artists-7c0045c40bac). 

This method basically divides the area into a grid, and calculate how dense the smoke is in each grid. Then calculate a divergence value(amount of smoke going in and out of this block) and pressure value based on the density of the adjacent blocks, then try to make the amount of smoke going in and out as equal as possible so that the smoke is not compressed anywhere. It is like solving a set NxM equations with NxM variables. To solve this massive equation as fast as possible, we can use the the parallel processing power of the GPU and apply the [Gauss-Seidel Method](https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method) to approximate the solution since it doesn't have to be 100% accurate for games.

However the Gauss-Seidel Method cannot be directly applied on the GPU side, because the neighboring blocks are being updated at the same time when the values are being used, so I had to apply the Red-Black Gauss-Seidel Method which was mentioned in the [second video](https://www.youtube.com/watch?v=Q78wvrQ9xsU&t=1856s). The Gauss-Seidel kernel code:
```GLSL
#pragma kernel SolveIncompressibility
[numthreads(8,8,1)]
void SolveIncompressibility (uint2 id : SV_DispatchThreadID)
{
    uint oddOrEven = id.y%2;
    int offset = (int)((redOrBlack+oddOrEven)%2);
    int2 coord = int2((int)(id.x)*2+offset, (int)id.y);
    if (!canFlow(coord)) return;
    int l = canFlow(coord+left);
    int r = canFlow(coord+right);
    int u = canFlow(coord+up);
    int d = canFlow(coord+down);
    int s = l+r+u+d;
    if (s == 0) return;
    float divergence = Gas[coord+right].x*r-Gas[coord].x +
        Gas[coord+up].y*u-Gas[coord].y;
    float pressure = ((Gas[coord+right].z*r+Gas[coord+left].z*l +
        Gas[coord+up].z*u+Gas[coord+down].z*d)- divergence)/s;
    float oldPressure = Gas[coord].z;
    float overPressure = oldPressure + (pressure-oldPressure)*overRelaxFactor;
    Gas[coord] = float3(Gas[coord].xy, overPressure);
}
```

Now I have a smoke simulation:

{% include video.liquid path="assets/video/smoke.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay: false%}

Then I have to add the smoke simulation to the previous demo, let water and solid interact with the smoke. This can easily be done by integrating the SDF map and the water density map we generated earlier into the smoke simulation and use it as the boundary map.

```GLSL

#define canFlow(pos) ((BoundaryMap[pos]))

#pragma kernel BuildBoundary
[numthreads(8,8,1)]
void BuildBoundary (uint2 id : SV_DispatchThreadID)
{
    BoundaryMap[id] = !fluidMap[id] && pixelsMap[id] > 0;
    if (BoundaryMap[id] ==0)
        SmokeMap[id] = 0;
}

#pragma kernel SetSmoke
[numthreads(8,8,1)]
void SetSmoke (uint2 id : SV_DispatchThreadID)
{
    int2 coord = (int2)id;
    int l = canFlow(coord+left);
    int r = canFlow(coord+right);
    int u = canFlow(coord+up);
    int d = canFlow(coord+down);
    int smoke = distance(id, smokePos) <= smokeRadius ? 1 : 0;
    if (smoke && l&r&u&d)
    {
        Gas[id] += float3(0, 1.5, 0);
        SmokeMap[id] = smokeColor*2;
    }
    else
    {
        Gas[id] *= float3(0.99,0.99,1);
        SmokeMap[id] *= 0.99;
    }
}

```

When advecting smoke and velocity, the code checks if it's outside of the boundary.

After this, I have a basic system simulating water, smoke, powder and solid in the same area:

{% include video.liquid path="assets/video/final_result.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay: false%}

The system still has much more to improve, and I still want to add a lot of features such as rigidbody interaction and chemical reactions. Maybe I can even extend it to 3D. The source code is available on [Github](https://github.com/miradilabk/2DPixelPhysics)

I started without knowing anything about fluid simulation or compute shaders. What a journey! Thanks for reading!