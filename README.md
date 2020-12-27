[![Build Status](https://travis-ci.org/MeirBon/l3d.svg?branch=main)](https://travis-ci.org/MeirBon/l3d)

# Loader-3d
A simple 3d loader written in Rust.
This project does not attempt to become the fastest loading 3D library, it serves as a simple library to quickly get started with the interesting (fun) stuff.
Currently, the project supports Obj and glTF files but adding more formats should be easy.

## Usage
```rust
use l3d::prelude::*;

// Create an instance
let instance = LoadInstance::new()
    // Adds default loaders (gLTF and obj files)
    .with_default();

// Load file
match instance.load(LoadOptions {
    path: PathBuf::from("path/to/my/file.gltf.obj"),
    ..Default::default()
}) {
    // Single mesh
    Mesh(descriptor) => {
        // descriptor.vertices: Vec<[f32; 4]>,
        // descriptor.normals: Vec<[f32; 3]>,
        // descriptor.uvs: Vec<[f32; 2]>,
        // descriptor.tangents: Vec<[f32; 4]>,
        // descriptor.material_ids: Vec<i32>,
        // --- Mesh descriptors do not have a material list when they are part of a scene ---
        // descriptor.materials: Option<MaterialList>,
        // descriptor.meshes: Vec<VertexMesh>,
        // descriptor.skeleton: Option<SkeletonDescriptor>,
        // descriptor.bounds: AABB,
        // descriptor.name: String,
    }
    // A scene with meshes and a scene graph
    Scene(descriptor) =>  {
        // descriptor.materials: MaterialList,
        // descriptor.meshes: Vec<MeshDescriptor>,
        // descriptor.nodes: Vec<NodeDescriptor>,
        // descriptor.animations: Vec<AnimationDescriptor>,
    }
    // Error
    None(LoadError),
}
```
