use super::{LoadOptions, MeshDescriptor};
use crate::{
    load::Loader,
    mat::Flip,
    mat::{MaterialList, Texture, TextureDescriptor, TextureSource},
    LoadError, LoadResult,
};
use glam::*;
use std::path::PathBuf;

#[derive(Debug, Copy, Clone)]
pub struct ObjLoader {}

impl std::fmt::Display for ObjLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "obj-loader")
    }
}

impl Default for ObjLoader {
    fn default() -> Self {
        Self {}
    }
}

impl Loader for ObjLoader {
    fn name(&self) -> &'static str {
        "obj loader"
    }

    fn file_extensions(&self) -> Vec<String> {
        vec![String::from("obj")]
    }

    fn load(&self, options: LoadOptions) -> LoadResult {
        let object = tobj::load_obj(&options.path);
        if let Err(_) = object {
            return LoadResult::None(LoadError::FileDoesNotExist(options.path.clone()));
        }

        let (models, materials) = object.unwrap();
        let mut material_indices: Vec<i32> = vec![-1; materials.len()];
        let mut mat_manager = MaterialList::new();

        for (i, material) in materials.iter().enumerate() {
            let mut color = Vec3::from(material.diffuse);
            let specular = Vec3::from(material.specular);

            let roughness = (1.0 - material.shininess.log10() / 1000.0)
                .max(0.0)
                .min(1.0);
            let opacity = 1.0 - material.dissolve;
            let eta = material.optical_density;

            let parent = if let Some(p) = options.path.parent() {
                p.to_path_buf()
            } else {
                PathBuf::new()
            };

            let d_path = if material.diffuse_texture == "" {
                None
            } else {
                Some(parent.join(material.diffuse_texture.as_str()).to_path_buf())
            };
            let mut n_path = if material.normal_texture == "" {
                None
            } else {
                Some(parent.join(material.normal_texture.as_str()).to_path_buf())
            };

            let mut roughness_map: Option<PathBuf> = None;
            let mut metallic_map: Option<PathBuf> = None;
            let mut emissive_map: Option<PathBuf> = None;
            let mut sheen_map: Option<PathBuf> = None;

            // TODO: Alpha and specular maps
            material.unknown_param.iter().for_each(|(name, value)| {
                let key = name.to_lowercase();
                match key.as_str() {
                    "ke" => {
                        // Emissive
                        let values = value.split_ascii_whitespace();
                        let mut f_values = [0.0 as f32; 3];
                        let mut i = 0;

                        for value in values {
                            assert!(i <= 2);
                            let value: f32 = value.parse().unwrap_or(0.0);
                            f_values[i] = value;
                            i += 1;
                        }

                        let mut value: Vec3A = Vec3A::from(f_values);
                        if !value.cmpeq(Vec3A::zero()).all() && value.cmple(Vec3A::one()).all() {
                            value = value * Vec3A::splat(10.0);
                        }

                        color = value.max(color.into()).into();
                    }
                    "map_pr" => {
                        roughness_map = Some(parent.join(value.as_str()).to_path_buf());
                    }
                    "map_ke" => {
                        emissive_map = Some(parent.join(value.as_str()).to_path_buf());
                    }
                    "ps" | "map_ps" => {
                        sheen_map = Some(parent.join(value.as_str()).to_path_buf());
                    }
                    "pm" | "map_pm" => {
                        metallic_map = Some(parent.join(value.as_str()).to_path_buf());
                    }
                    "norm" | "map_ns" | "map_bump" => {
                        n_path = Some(parent.join(value.as_str()).to_path_buf());
                    }
                    _ => {}
                }
            });

            let metallic_roughness = match (roughness_map, metallic_map) {
                (Some(r), Some(m)) => {
                    let r = match Texture::load(&r, Flip::FlipV) {
                        Ok(t) => t,
                        Err(_) => return LoadResult::None(LoadError::TextureDoesNotExist(r)),
                    };
                    let m = match Texture::load(&m, Flip::FlipV) {
                        Ok(t) => t,
                        Err(_) => return LoadResult::None(LoadError::TextureDoesNotExist(m)),
                    };

                    let (r, m) = if r.width != m.width || r.height != m.height {
                        let width = r.width.max(m.width);
                        let height = r.height.max(m.height);
                        (r.resized(width, height), m.resized(width, height))
                    } else {
                        (r, m)
                    };

                    let combined = Texture::merge(Some(&r), Some(&m), None, None);
                    Some(TextureSource::Loaded(combined))
                }
                (Some(r), None) => {
                    let r = match Texture::load(&r, Flip::FlipV) {
                        Ok(t) => t,
                        Err(_) => return LoadResult::None(LoadError::TextureDoesNotExist(r)),
                    };

                    let combined = Texture::merge(Some(&r), None, None, None);
                    Some(TextureSource::Loaded(combined))
                }
                (None, Some(m)) => {
                    let m = match Texture::load(&m, Flip::FlipV) {
                        Ok(t) => t,
                        Err(_) => return LoadResult::None(LoadError::TextureDoesNotExist(m)),
                    };
                    let combined = Texture::merge(None, Some(&m), None, None);
                    Some(TextureSource::Loaded(combined))
                }
                _ => None,
            };

            let mat_index = mat_manager.add_with_maps(
                color,
                roughness,
                specular,
                opacity,
                TextureDescriptor {
                    albedo: if let Some(path) = d_path {
                        Some(TextureSource::Filesystem(path, Flip::FlipV))
                    } else {
                        None
                    },
                    normal: if let Some(path) = n_path {
                        Some(TextureSource::Filesystem(path, Flip::FlipV))
                    } else {
                        None
                    },
                    metallic_roughness_map: metallic_roughness,
                    emissive_map: if let Some(path) = emissive_map {
                        Some(TextureSource::Filesystem(path, Flip::FlipV))
                    } else {
                        None
                    },
                    sheen_map: if let Some(path) = sheen_map {
                        Some(TextureSource::Filesystem(path, Flip::FlipV))
                    } else {
                        None
                    },
                },
            );

            mat_manager[mat_index].eta = eta;
            material_indices[i] = mat_index as i32;
        }

        if material_indices.is_empty() {
            material_indices.push(-1);
        }

        let num_vertices: usize = models.iter().map(|m| m.mesh.indices.len()).sum();

        let mut vertices: Vec<[f32; 4]> = Vec::with_capacity(num_vertices);
        let mut normals: Vec<[f32; 3]> = Vec::with_capacity(num_vertices);
        let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(num_vertices);
        let mut material_ids: Vec<i32> = Vec::with_capacity(num_vertices);

        for m in models.iter() {
            let mesh = &m.mesh;

            let mut i = 0;
            for idx in &mesh.indices {
                let idx = *idx as usize;
                let i0 = 3 * idx;
                let i1 = i0 + 1;
                let i2 = i0 + 2;

                let pos = [
                    mesh.positions[i0],
                    mesh.positions[i1],
                    mesh.positions[i2],
                    1.0,
                ];

                let normal = if !mesh.normals.is_empty() {
                    [mesh.normals[i0], mesh.normals[i1], mesh.normals[i2]]
                } else {
                    [0.0; 3]
                };

                let uv = if !mesh.texcoords.is_empty() {
                    [mesh.texcoords[idx * 2], mesh.texcoords[idx * 2 + 1]]
                } else {
                    [0.0; 2]
                };

                vertices.push(pos);
                normals.push(normal);
                uvs.push(uv);

                let material_id = if mesh.material_id.is_some() {
                    (*material_indices
                        .get(mesh.material_id.unwrap())
                        .unwrap_or(&0)) as i32
                } else {
                    material_indices[0] as i32
                };

                material_ids.push(material_id);
                i = i + 1;
            }
        }

        let descriptor = MeshDescriptor::new(
            vertices,
            normals,
            uvs,
            None,
            material_ids,
            Some(mat_manager),
            None,
        );

        LoadResult::Mesh(descriptor)
    }
}

#[cfg(test)]
mod tests {
    use rtbvh::AABB;

    use super::Loader;
    use super::ObjLoader;
    use crate::load::LoadOptions;
    use core::panic;
    use std::path::PathBuf;

    #[test]
    fn load_obj_works() {
        let loader = ObjLoader::default();
        let sphere = loader.load(LoadOptions {
            path: PathBuf::from("assets/sphere.obj"),
            ..Default::default()
        });

        let m = match sphere {
            crate::LoadResult::Mesh(m) => m,
            crate::LoadResult::Scene(_) => panic!("Obj loader should only return meshes"),
            crate::LoadResult::None(_) => panic!("Obj loader should successfully load meshes"),
        };

        // Bounds should be correct
        let mut aabb = AABB::new();
        for v in m.vertices.iter() {
            aabb.grow([v[0], v[1], v[2]]);
        }
        for i in 0..3 {
            assert_eq!(aabb.min[i], m.bounds.min[i]);
            assert_eq!(aabb.max[i], m.bounds.max[i]);
        }

        assert_eq!(960 * 3, m.vertices.len(), "The sphere object has 960 faces");
        assert_eq!(
            m.vertices.len(),
            m.normals.len(),
            "Number of vertices and normals should be equal"
        );
        assert_eq!(
            m.vertices.len(),
            m.uvs.len(),
            "Number of vertices and uvs should be equal"
        );
        assert_eq!(
            m.vertices.len(),
            m.tangents.len(),
            "Number of vertices and tangents should be equal"
        );
        assert_eq!(
            m.vertices.len(),
            m.material_ids.len(),
            "Number of vertices and material ids should be equal"
        );
    }

    #[test]
    fn load_obj_with_options_works() {
        // TODO: This does not work yet
        let loader = ObjLoader::default();
        let sphere = loader.load(LoadOptions {
            path: PathBuf::from("assets/sphere.obj"),
            with_normals: false,
            with_tangents: false,
            with_materials: false,
        });

        let m = match sphere {
            crate::LoadResult::Mesh(m) => m,
            crate::LoadResult::Scene(_) => panic!("Obj loader should only return meshes"),
            crate::LoadResult::None(_) => panic!("Obj loader should succesfully load meshes"),
        };

        // Bounds should be correct
        let mut aabb = AABB::new();
        for v in m.vertices.iter() {
            aabb.grow([v[0], v[1], v[2]]);
        }
        for i in 0..3 {
            assert_eq!(aabb.min[i], m.bounds.min[i]);
            assert_eq!(aabb.max[i], m.bounds.max[i]);
        }

        assert_eq!(960 * 3, m.vertices.len(), "The sphere object has 960 faces");
        assert_eq!(
            m.vertices.len(),
            m.normals.len(),
            "Number of vertices and normals should be equal"
        );
        assert_eq!(
            m.vertices.len(),
            m.uvs.len(),
            "Number of vertices and uvs should be equal"
        );
        assert_eq!(
            m.vertices.len(),
            m.tangents.len(),
            "Number of vertices and tangents should be equal"
        );
        assert_eq!(
            m.vertices.len(),
            m.material_ids.len(),
            "Number of vertices and material ids should be equal"
        );
    }
}
