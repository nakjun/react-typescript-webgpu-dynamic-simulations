# Real-Time Mass-Spring Simulation System using WebGPU Frameworks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14186494.svg)](https://doi.org/10.5281/zenodo.14186494)

### Setup & Run Projects
```shell
$npm install #install npm dependencies(react, webgpu, ...)
$npm run start #start react app
#Open https://localhost:3000 to view it in the browser.
```

## Features
- Mass-Spring System : Spring-centric method without Self-Collision
- Collision System : [Detection Part] AABB-based BVH(Broad Phase) + Tri-Tri Intersection(Narrow Phase) / [Response Part] Triangle-Repsonse Method
- Rendering System : WebGPU Renderer
- 3D Surface models
    - Sphere : 0.4K Vertices, 0.9K Triangles(Faces)
    - Armadillo : 25.3K Vertices, 50.6K Triangles(Faces)
    - Dragon   : 50K Vertices, 100K Triangles(Faces)


## Experimental Videos
[![videos](https://img.youtube.com/vi/AXY6gcJpZYQ/0.jpg)](https://youtu.be/AXY6gcJpZYQ)

## Cite This Projects
※ This project is currently being prepared for submission to a peer-reviewed journal.
```bibtex
@misc{ClothSimulationWebGPU,
  author       = {Nak-Jun Sung},
  title        = {Real-Time Cloth Simulation Using WebGPU: Evaluating Limits of High-Resolution},  
  year         = {2024},  
  doi          = {https://doi.org/10.5281/zenodo.14186494}
}
```