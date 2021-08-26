# NBodies
A C# &amp; OpenCL 2D SPH NBody Simulation Toy. Utilizes a compressed multi-level hybrid fast multipole "mesh" for a high performance loosely bound simulation.

Features
----------

* Loosely bound field: Supports very large complex and sparse simulations. Multiple passes will be used where allocation size exceeds the max allowed by the platform.
* Hybrid fast multipole "mesh" optimization: A multi-layer mesh and neighbor list is constructed using optimized hybrid CPU/GPU methods. And aliasing is reduced by averaging each mesh cells center of mass. Cell size and level count can be adjusted at runtime.
* Multi-mode collisional physics: Bodies collide with SPH or elastic/merge collisions depending on their size & roche status. Collisions can also be disabled completely.
* Large body roche fractures: Large bodies will fracture into a circular clump of particles when the gravitational force exceeds a ratio of thier mass.
* Save States: Fields and be saved and loaded from disk via a ProtoBuf serializer.
* Simple User Interaction: Fields/bodies can be added, moved and edited on the fly with simple mouse & keyboard inputs. Navigation around the field is done via intuitive click-drag Google Maps style inputs.
* Multiple Display Styles & Scaling: Normal, Pressure, Density, Velocity, Index, Spatial Order, Force and High Contrast display types with scaling that can be changed at runtime.
* Rewind Buffer: The option to selectively rewind the simulation up to 200 frames can be enabled.
* Dual-mode abstracted rendering: The default renderer utilizes SharpDX for Direct 2D rendering. But a GDI based renderer is included and they can be swiched anytime during runtime.
* Matter types with density: A basic series of matter types with varying densities can be used to create complex SPH interactions including stratification of clumps.
* etc, etc, etc

Performance
-----------

* Intel i7-6820HQ
* AMD Radeon R7 M370

(Varies with field density and clumps)

##### 5k particles #####
250+ FPS

##### 10k particles #####
~150 FPS

##### 20k particles #####
~100 FPS

##### 50k particles #####
60 - 70 FPS

##### 100k particles #####
40 - 50 FPS

##### 400k particles #####
~14 FPS

