# RayTracer
A ray tracer written using both the Wolfram Language and CUDA/C++.

A Mathematica notebook is used as the frontend for specifying the scene and render settings (objects, lighting, materials, resolution, etc.). This information is then passed to a CUDA kernel using <a href="https://reference.wolfram.com/language/CUDALink/tutorial/Overview.html">CUDALink</a>. The resulting pixels are then sent back to the notebook.

This method combines the flexibility of the Wolfram Language with the performance of GPU programming.

## Features
- BVH Acceleration Structure (can support many objects)
- Perfect Reflections (i.e. mirrors)
- Perfect Transmission (i.e. glass)
- Directional and Point Lights
- Spherical Skybox
- Adjustable Anti-Aliasing
- Orthographic and Perspective Camera

## Examples

<img src="https://github.com/Shedelbower/RayTracer/blob/master/Renders/bunny.png?raw=true">

<img src="https://github.com/Shedelbower/RayTracer/blob/master/Renders/mirror.png?raw=true">

<img src="https://github.com/Shedelbower/RayTracer/blob/master/Renders/pumpkin_patch.jpg?raw=true">

<img src="https://github.com/Shedelbower/RayTracer/blob/master/Renders/teapot_skybox.png?raw=true">

<img src="https://github.com/Shedelbower/RayTracer/blob/master/Renders/Animated/wavey_spheres.gif?raw=true">

<img src="https://github.com/Shedelbower/RayTracer/blob/master/Renders/Animated/zebra_texture.gif?raw=true">
