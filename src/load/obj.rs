use super::{LoadOptions, MeshDescriptor};
use crate::{
    load::Loader,
    mat::Flip,
    mat::{MaterialList, Texture, TextureDescriptor, TextureSource},
    LoadError, LoadResult,
};
use glam::*;
use std::{convert::TryFrom, fs::File, path::PathBuf};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
        let obj_options = tobj::LoadOptions {
            single_index: true,
            triangulate: true,
            ignore_points: false,
            ignore_lines: false,
        };

        let parent = match &options.source {
            crate::load::LoadSource::Path(p) => {
                p.parent().map(|f| f.to_path_buf()).unwrap_or_default()
            }
            crate::load::LoadSource::String { basedir, .. } => {
                PathBuf::try_from(basedir).unwrap_or_default()
            }
        };

        let object = match &options.source {
            crate::load::LoadSource::Path(p) => tobj::load_obj(&p, &obj_options),
            crate::load::LoadSource::String { source, .. } => {
                use std::io::BufReader;
                tobj::load_obj_buf(&mut BufReader::new(*source), &obj_options, |p| {
                    if let Some(f) = p.file_name().and_then(|f| f.to_str()) {
                        let f = if let Ok(f) = File::open(parent.join(PathBuf::from(f))) {
                            f
                        } else {
                            return tobj::MTLLoadResult::Err(tobj::LoadError::ReadError);
                        };
                        tobj::load_mtl_buf(&mut BufReader::new(f))
                    } else {
                        tobj::MTLLoadResult::Err(tobj::LoadError::ReadError)
                    }
                })
            }
        };

        if object.is_err() {
            match &options.source {
                crate::load::LoadSource::Path(p) => {
                    return LoadResult::None(LoadError::FileDoesNotExist(p.clone()))
                }
                crate::load::LoadSource::String { .. } => {
                    return LoadResult::None(LoadError::CouldNotParseSource)
                }
            }
        }

        let (models, materials) = object.unwrap();
        let materials = materials.unwrap_or_default();
        let mut material_indices: Vec<i32> = vec![-1; materials.len()];
        let mut mat_manager = MaterialList::new();

        for (i, material) in materials.iter().enumerate() {
            let mut color = Vec3::from(material.diffuse);
            let specular = Vec3::from(material.specular);

            let roughness = (1.0_f32 - material.shininess.log10() / 1000.0_f32).clamp(0.0, 1.0);
            let opacity = 1.0 - material.dissolve;
            let eta = material.optical_density;

            let d_path = if material.diffuse_texture.is_empty() {
                None
            } else {
                Some(parent.join(material.diffuse_texture.as_str()).to_path_buf())
            };
            let mut n_path = if material.normal_texture.is_empty() {
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
                        let mut f_values = [0.0_f32; 3];

                        for (i, value) in values.take(3).enumerate() {
                            let value: f32 = value.parse().unwrap_or(0.0);
                            f_values[i] = value;
                        }

                        let mut value: Vec3A = Vec3A::from(f_values);
                        if !value.cmpeq(Vec3A::ZERO).all() && value.cmple(Vec3A::ONE).all() {
                            value *= Vec3A::splat(10.0);
                        }

                        color = value.max(color.into()).into();
                    }
                    "map_pr" => {
                        roughness_map = Some(parent.join(value.as_str()));
                    }
                    "map_ke" => {
                        emissive_map = Some(parent.join(value.as_str()));
                    }
                    "ps" | "map_ps" => {
                        sheen_map = Some(parent.join(value.as_str()));
                    }
                    "pm" | "map_pm" => {
                        metallic_map = Some(parent.join(value.as_str()));
                    }
                    "norm" | "map_ns" | "map_bump" => {
                        n_path = Some(parent.join(value.as_str()));
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
                    albedo: d_path.map(|path| TextureSource::Filesystem(path, Flip::FlipV)),
                    normal: n_path.map(|path| TextureSource::Filesystem(path, Flip::FlipV)),
                    metallic_roughness_map: metallic_roughness,
                    emissive_map: emissive_map
                        .map(|path| TextureSource::Filesystem(path, Flip::FlipV)),
                    sheen_map: sheen_map.map(|path| TextureSource::Filesystem(path, Flip::FlipV)),
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
    use super::*;
    use crate::load::*;
    use core::panic;
    use rtbvh::Aabb;
    use std::path::PathBuf;

    #[test]
    fn load_obj_works() {
        let loader = ObjLoader::default();
        let sphere = loader.load(LoadOptions {
            source: LoadSource::Path(PathBuf::from("assets/sphere.obj")),
            ..Default::default()
        });

        let m = match sphere {
            crate::LoadResult::Mesh(m) => m,
            crate::LoadResult::Scene(_) => panic!("Obj loader should only return meshes"),
            crate::LoadResult::None(_) => panic!("Obj loader should successfully load meshes"),
        };

        // Bounds should be correct
        let mut aabb: Aabb<()> = Aabb::default();
        for v in m.vertices.iter() {
            aabb.grow(vec3(v[0], v[1], v[2]));
        }
        for i in 0..3 {
            assert!((aabb.min[i] - m.bounds.min[i]).abs() < f32::EPSILON);
            assert!((aabb.max[i] - m.bounds.max[i]).abs() < f32::EPSILON);
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
    fn load_obj_source_works() {
        let loader = ObjLoader::default();
        let sphere = loader.load(LoadOptions {
            source: LoadSource::String {
                source: include_bytes!("../../assets/sphere.obj"),
                extension: "obj",
                basedir: "assets",
            },
            ..Default::default()
        });

        let m = match sphere {
            crate::LoadResult::Mesh(m) => m,
            crate::LoadResult::Scene(_) => panic!("Obj loader should only return meshes"),
            crate::LoadResult::None(_) => panic!("Obj loader should successfully load meshes"),
        };

        // Bounds should be correct
        let mut aabb: Aabb<()> = Aabb::default();
        for v in m.vertices.iter() {
            aabb.grow(vec3(v[0], v[1], v[2]));
        }
        for i in 0..3 {
            assert!((aabb.min[i] - m.bounds.min[i]).abs() < f32::EPSILON);
            assert!((aabb.max[i] - m.bounds.max[i]).abs() < f32::EPSILON);
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
            source: LoadSource::Path(PathBuf::from("assets/sphere.obj")),
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
        let mut aabb: Aabb<()> = Aabb::new();
        for v in m.vertices.iter() {
            aabb.grow(vec3(v[0], v[1], v[2]));
        }
        for i in 0..3 {
            assert!((aabb.min[i] - m.bounds.min[i]).abs() < f32::EPSILON);
            assert!((aabb.max[i] - m.bounds.max[i]).abs() < f32::EPSILON);
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
