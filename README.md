# RT 01

A stochastic path tracer built for my own learning.

Much of the code is hard-wired to render a Mercedes Benz 300sl animation. You can view it [here](https://www.youtube.com/1234).

Features:
- Normal mapping, displacement to normals
- Normal ray plane clamping
- Pseudo HDR sky using Perlin Noise
- Depth blur
- Motion blur
- Per pixel sample stratification for smooth motion and depth blur
- Uses Intel's OIDN for nice noise free output (even 16 rpp gave reasonable results)

Learning resources:
- Ray Tracing Minibooks, Pete Shirley
- Realtime Ray Tracing and Interactive Global Illumination, Wald
- Ray-Triangle Intersection, Möller–Trumbore
- Wikipedia: Cramer’s rule
- On building fast kd-Trees for Ray Tracing, Wald-Havran
- Mathematics for 3D Game Programming and Computer Graphics

I would _not_ recommend using this for any serious projects. Most of the code is not that efficient. However, I was satisfied, given this was my first attempt at making a renderer that could draw more than just spheres.

Since the Mercedes model is copyrighted it has not been committed to this repository.
