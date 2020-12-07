use crate::{mat::MaterialList, LoadResult};
use glam::*;
use rtbvh::{Bounds, AABB};
use std::{
    fmt::{Debug, Display},
    path::PathBuf,
};

pub mod gltf;
pub mod obj;

pub trait Loader: Debug {
    fn name(&self) -> &'static str;
    fn file_extensions(&self) -> Vec<String>;
    fn load(&self, options: LoadOptions) -> LoadResult;
}

pub struct LoadOptions {
    pub path: PathBuf,
    /// Whether to load/generate normals if they are not part of the 3D file
    pub with_normals: bool,
    /// Whether to load/generate tangents if they are not part of the 3D file
    pub with_tangents: bool,
    /// Whether to load materials
    pub with_materials: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            with_normals: true,
            with_tangents: true,
            with_materials: true,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct VertexMesh {
    pub first: u32,
    pub last: u32,
    pub mat_id: i32,
    pub bounds: AABB,
}

impl Display for VertexMesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VertexMesh {{ first: {}, last: {}, mat_id: {}, bounds: {} }}",
            self.first, self.last, self.mat_id, self.bounds
        )
    }
}

#[derive(Debug, Clone)]
pub struct SkinDescriptor {
    pub name: String,
    pub inverse_bind_matrices: Vec<[f32; 16]>,
    // Joint node descriptor IDs (NodeDescriptor::id)
    pub joint_nodes: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct AnimationDescriptor {
    pub name: String,
    // (node descriptor ID, animation channel)
    pub channels: Vec<(u32, Channel)>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Method {
    Linear,
    Spline,
    Step,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Target {
    Translation,
    Rotation,
    Scale,
    MorphWeights,
}

#[derive(Debug, Copy, Clone)]
pub struct Orthographic {
    pub x_mag: f32,
    pub y_mag: f32,
    pub z_near: f32,
    pub z_far: f32,
}

#[derive(Debug, Copy, Clone)]
pub struct Perspective {
    pub aspect_ratio: Option<f32>,
    pub y_fov: f32,
    pub z_near: f32,
    pub z_far: Option<f32>,
}

#[derive(Debug, Clone)]
pub enum Projection {
    Orthographic(Orthographic),
    Perspective(Perspective),
}

#[derive(Debug, Clone)]
pub struct CameraDescriptor {
    projection: Projection,
}

#[derive(Debug, Clone)]
pub struct NodeDescriptor {
    pub name: String,
    pub child_nodes: Vec<NodeDescriptor>,
    pub camera: Option<CameraDescriptor>,

    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],

    pub meshes: Vec<u32>,
    pub skin: Option<SkinDescriptor>,
    pub weights: Vec<f32>,

    /// An ID that is guaranteed to be unique within the scene descriptor this
    /// node descriptor belongs to.
    pub id: u32,
}

#[derive(Debug, Clone)]
pub struct Channel {
    pub targets: Vec<Target>,
    pub key_frames: Vec<f32>,

    pub sampler: Method,
    pub vec3s: Vec<[f32; 3]>,
    /// xyzw quaternion rotations
    pub rotations: Vec<[f32; 4]>,
    pub weights: Vec<f32>,

    pub duration: f32,
}

impl Default for Channel {
    fn default() -> Self {
        Self {
            targets: Vec::new(),
            key_frames: Vec::new(),

            sampler: Method::Linear,
            vec3s: Vec::new(),
            rotations: Vec::new(),
            weights: Vec::new(),

            duration: 0.0,
        }
    }
}

impl Channel {
    pub fn sample_translation(&self, time: f32, k: usize) -> [f32; 3] {
        let t0 = self.key_frames[k];
        let t1 = self.key_frames[k + 1];
        let f = (time - t0) / (t1 - t0);

        match self.sampler {
            Method::Linear => {
                ((1.0 - f) * Vec3::from(self.vec3s[k]) + f * Vec3::from(self.vec3s[k + 1])).into()
            }
            Method::Spline => {
                let t = f;
                let t2 = t * t;
                let t3 = t2 * t;
                let dt = t1 - t0;
                let p0 = Vec3::from(self.vec3s[k * 3 + 1]);
                let m0 = dt * Vec3::from(self.vec3s[k * 3 + 2]);
                let p1 = Vec3::from(self.vec3s[(k + 1) * 3 + 1]);
                let m1 = dt * Vec3::from(self.vec3s[(k + 1) * 3]);

                let result: Vec3 = m0 * (t3 - 2.0 * t2 + t)
                    + p0 * (2.0 * t3 - 3.0 * t2 + 1.0)
                    + p1 * (-2.0 * t3 + 3.0 * t2)
                    + m1 * (t3 - t2);
                result.into()
            }
            Method::Step => self.vec3s[k],
        }
    }

    pub fn sample_scale(&self, time: f32, k: usize) -> [f32; 3] {
        let t0 = self.key_frames[k];
        let t1 = self.key_frames[k + 1];
        let f = (time - t0) / (t1 - t0);

        match self.sampler {
            Method::Linear => {
                ((1.0 - f) * Vec3::from(self.vec3s[k]) + f * Vec3::from(self.vec3s[k + 1])).into()
            }
            Method::Spline => {
                let t = f;
                let t2 = t * t;
                let t3 = t2 * t;
                let dt = t1 - t0;
                let p0 = Vec3::from(self.vec3s[k * 3 + 1]);
                let m0 = dt * Vec3::from(self.vec3s[k * 3 + 2]);
                let p1 = Vec3::from(self.vec3s[(k + 1) * 3 + 1]);
                let m1 = dt * Vec3::from(self.vec3s[(k + 1) * 3]);
                let result = m0 * (t3 - 2.0 * t2 + t)
                    + p0 * (2.0 * t3 - 3.0 * t2 + 1.0)
                    + p1 * (-2.0 * t3 + 3.0 * t2)
                    + m1 * (t3 - t2);
                result.into()
            }
            Method::Step => self.vec3s[k],
        }
    }

    pub fn sample_weight(&self, time: f32, k: usize, i: usize, count: usize) -> f32 {
        let t0 = self.key_frames[k];
        let t1 = self.key_frames[k + 1];
        let f = (time - t0) / (t1 - t0);

        match self.sampler {
            Method::Linear => {
                (1.0 - f) * self.weights[k * count + i] + f * self.weights[(k + 1) * count + i]
            }
            Method::Spline => {
                let t = f;
                let t2 = t * t;
                let t3 = t2 * t;
                let p0 = self.weights[(k * count + i) * 3 + 1];
                let m0 = (t1 - t0) * self.weights[(k * count + i) * 3 + 2];
                let p1 = self.weights[((k + 1) * count + i) * 3 + 1];
                let m1 = (t1 - t0) * self.weights[((k + 1) * count + i) * 3];
                m0 * (t3 - 2.0 * t2 + t)
                    + p0 * (2.0 * t3 - 3.0 * t2 + 1.0)
                    + p1 * (-2.0 * t3 + 3.0 * t2)
                    + m1 * (t3 - t2)
            }
            Method::Step => self.weights[k],
        }
    }

    pub fn sample_rotation(&self, time: f32, k: usize) -> [f32; 4] {
        let t0 = self.key_frames[k];
        let t1 = self.key_frames[k + 1];
        let f = (time - t0) / (t1 - t0);

        match self.sampler {
            Method::Linear => ((Vec4::from(self.rotations[k]) * (1.0 - f))
                + (Vec4::from(self.rotations[k + 1]) * f))
                .into(),
            Method::Spline => {
                let t = f;
                let t2 = t * t;
                let t3 = t2 * t;
                let dt = t1 - t0;

                let p0 = Vec4::from(self.rotations[k * 3 + 1]);
                let m0 = Vec4::from(self.rotations[k * 3 + 2]) * dt;
                let p1 = Vec4::from(self.rotations[(k + 1) * 3 + 1]);
                let m1 = Vec4::from(self.rotations[(k + 1) * 3]) * dt;
                let result = m0 * (t3 - 2.0 * t2 + t)
                    + p0 * (2.0 * t3 - 3.0 * t2 + 1.0)
                    + p1 * (-2.0 * t3 + 3.0 * t2)
                    + m1 * (t3 - t2);
                result.into()
            }
            Method::Step => self.rotations[k],
        }
    }
}

#[allow(dead_code)]
#[cfg_attr(feature = "object_caching", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Animation {
    pub name: String,
    pub affected_roots: Vec<u32>,
    pub channels: Vec<(u32, Channel)>, // Vec<(node id, channel)>
}

impl Default for Animation {
    fn default() -> Self {
        Self {
            name: String::new(),
            affected_roots: Vec::new(),
            channels: Vec::new(),
        }
    }
}

#[allow(dead_code)]
impl Animation {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_time<T, R, S, Mw, M>(
        &mut self,
        time: f32,
        update_t: &mut T,
        update_r: &mut R,
        update_s: &mut S,
        update_mw: &mut Mw,
        update_matrix: &mut M,
    ) where
    // (Node id, translation)
        T: FnMut(usize, [f32; 3]),
    // (Node id, rotation)
        R: FnMut(usize, [f32; 4]),
    // (Node id, scale)
        S: FnMut(usize, [f32; 3]),
    // (Node id, morph index, morph weight)
        Mw: FnMut(usize, usize, f32),
    // (Node id), A function call signaling that the matrix of a node should be updated
        M: FnMut(usize),
    {
        let channels = &mut self.channels;

        channels.iter_mut().for_each(|(node_id, c)| {
            let current_time = time % c.duration;
            let node_id = *node_id as usize;
            c.targets.iter().for_each(|t| {
                let mut key = 0;
                while current_time > c.key_frames[key as usize + 1] {
                    key += 1;
                }

                match t {
                    Target::Translation => {
                        update_t(node_id, c.sample_translation(current_time, key));
                    }
                    Target::Rotation => {
                        update_r(node_id, c.sample_rotation(current_time, key));
                    }
                    Target::Scale => {
                        update_s(node_id, c.sample_scale(current_time, key));
                    }
                    Target::MorphWeights => {
                        let weights = c.weights.len();
                        for i in 0..weights {
                            update_mw(node_id, i, c.sample_weight(current_time, key, i, weights));
                        }
                    }
                }
            });

            update_matrix(node_id);
        });
    }
}

#[derive(Debug, Clone)]
pub struct SceneDescriptor {
    pub materials: MaterialList,
    pub meshes: Vec<MeshDescriptor>,
    pub nodes: Vec<NodeDescriptor>,
    pub animations: Vec<AnimationDescriptor>,
}

use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct SkeletonDescriptor {
    pub joints: Vec<Vec<[u16; 4]>>,
    pub weights: Vec<Vec<[f32; 4]>>,
}

#[derive(Debug, Clone)]
pub struct MeshDescriptor {
    pub vertices: Vec<[f32; 4]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub tangents: Vec<[f32; 4]>,
    pub material_ids: Vec<i32>,
    /// Mesh descriptors do not have a material list when they are part of a scene
    pub materials: Option<MaterialList>,
    pub meshes: Vec<VertexMesh>,
    pub skeleton: Option<SkeletonDescriptor>,
    pub bounds: AABB,
    pub name: String,
}

impl Display for MeshDescriptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Mesh {{ vertices: {}, materials: {}, meshes: {}, bounds: {}, name: {} }}",
            self.vertices.len(),
            if let Some(m) = self.materials.as_ref() {
                m.len()
            } else {
                0
            },
            self.meshes.len(),
            self.bounds,
            self.name.as_str()
        )
    }
}

impl Default for MeshDescriptor {
    fn default() -> Self {
        MeshDescriptor::empty()
    }
}

impl MeshDescriptor {
    pub fn new_indexed(
        indices: Vec<[u32; 3]>,
        original_vertices: Vec<[f32; 4]>,
        original_normals: Vec<[f32; 3]>,
        original_uvs: Vec<[f32; 2]>,
        skeleton: Option<SkeletonDescriptor>,
        material_ids: Vec<i32>,
        materials: Option<MaterialList>,
        name: Option<String>,
    ) -> Self {
        let mut vertices: Vec<[f32; 4]> = Vec::with_capacity(indices.len() * 3);
        let mut normals: Vec<[f32; 3]> = Vec::with_capacity(indices.len() * 3);
        let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(indices.len() * 3);
        let mut material_indices: Vec<i32> = Vec::with_capacity(indices.len() * 3);

        let (mut joints, org_joints, mut weights, org_weights) = if let Some(skeleton) = skeleton {
            let mut joints = Vec::with_capacity(skeleton.joints.len());
            let mut weights = Vec::with_capacity(skeleton.weights.len());

            for j in skeleton.joints.iter() {
                joints.push(Vec::with_capacity(j.len()));
            }
            for w in skeleton.weights.iter() {
                weights.push(Vec::with_capacity(w.len()));
            }

            (joints, skeleton.joints, weights, skeleton.weights)
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };

        indices.into_iter().enumerate().for_each(|(j, i)| {
            let i0 = i[0] as usize;
            let i1 = i[1] as usize;
            let i2 = i[2] as usize;

            vertices.push([
                original_vertices[i0][0],
                original_vertices[i0][1],
                original_vertices[i0][2],
                1.0,
            ]);
            vertices.push([
                original_vertices[i1][0],
                original_vertices[i1][1],
                original_vertices[i1][2],
                1.0,
            ]);
            vertices.push([
                original_vertices[i2][0],
                original_vertices[i2][1],
                original_vertices[i2][2],
                1.0,
            ]);

            normals.push([
                original_normals[i0][0],
                original_normals[i0][1],
                original_normals[i0][2],
            ]);
            normals.push([
                original_normals[i1][0],
                original_normals[i1][1],
                original_normals[i1][2],
            ]);
            normals.push([
                original_normals[i2][0],
                original_normals[i2][1],
                original_normals[i2][2],
            ]);

            uvs.push([original_uvs[i0][0], original_uvs[i0][1]]);
            uvs.push([original_uvs[i1][0], original_uvs[i1][1]]);
            uvs.push([original_uvs[i2][0], original_uvs[i2][1]]);

            joints.iter_mut().enumerate().for_each(|(i, v)| {
                v.push(org_joints[i][i0]);
                v.push(org_joints[i][i1]);
                v.push(org_joints[i][i2]);
            });

            weights.iter_mut().enumerate().for_each(|(i, v)| {
                v.push(org_weights[i][i0]);
                v.push(org_weights[i][i1]);
                v.push(org_weights[i][i2]);
            });

            material_indices.push(material_ids[j]);
            material_indices.push(material_ids[j + 1]);
            material_indices.push(material_ids[j + 2]);
        });

        let skeleton = if !joints.is_empty() && !weights.is_empty() {
            Some(SkeletonDescriptor { joints, weights })
        } else {
            None
        };

        debug_assert_eq!(vertices.len(), normals.len());
        debug_assert_eq!(vertices.len(), uvs.len());
        debug_assert_eq!(uvs.len(), material_ids.len());
        debug_assert_eq!(vertices.len() % 3, 0);

        Self::new(
            vertices,
            normals,
            uvs,
            skeleton,
            material_indices,
            materials,
            name,
        )
    }

    pub fn new(
        vertices: Vec<[f32; 4]>,
        normals: Vec<[f32; 3]>,
        uvs: Vec<[f32; 2]>,
        skeleton: Option<SkeletonDescriptor>,
        material_ids: Vec<i32>,
        materials: Option<MaterialList>,
        name: Option<String>,
    ) -> Self {
        debug_assert_eq!(vertices.len(), normals.len());
        debug_assert_eq!(vertices.len(), uvs.len());
        debug_assert_eq!(uvs.len(), material_ids.len());
        debug_assert_eq!(vertices.len() % 3, 0);

        let mut bounds = AABB::new();

        // Generate normals
        let normals: Vec<[f32; 3]> = if Vec3::from(normals[0]).cmpeq(Vec3::zero()).all() {
            let mut normals = vec![[0.0_f32; 3]; vertices.len()];
            for i in (0..vertices.len()).step_by(3) {
                let v0 = Vec3::new(vertices[i + 0][0], vertices[i + 0][1], vertices[i + 0][2]);
                let v1 = Vec3::new(vertices[i + 1][0], vertices[i + 1][1], vertices[i + 1][2]);
                let v2 = Vec3::new(vertices[i + 2][0], vertices[i + 2][1], vertices[i + 2][2]);

                let e1: Vec3 = v1 - v0;
                let e2: Vec3 = v2 - v0;

                let n = e1.cross(e2).normalize();

                let a = (v1 - v0).length();
                let b = (v2 - v1).length();
                let c = (v0 - v2).length();
                let s = (a + b + c) * 0.5;
                let area = (s * (s - a) * (s - b) * (s - c)).sqrt();
                let n: Vec3 = n * area;

                for j in 0..3 {
                    normals[i + 0][j] += n[j];
                    normals[i + 1][j] += n[j];
                    normals[i + 2][j] += n[j];
                }
            }

            normals
                .par_iter_mut()
                .for_each(|n| *n = Vec3::from(*n).normalize().into());
            normals
        } else {
            normals
        };

        // Generate tangents
        let mut tangents: Vec<[f32; 4]> = vec![[0.0_f32; 4]; vertices.len()];
        let mut bitangents: Vec<[f32; 3]> = vec![[0.0_f32; 3]; vertices.len()];

        for i in (0..vertices.len()).step_by(3) {
            let v0: Vec3 = Vec3::new(vertices[i][0], vertices[i][1], vertices[i][2]);
            let v1: Vec3 = Vec3::new(vertices[i + 1][0], vertices[i + 1][1], vertices[i + 1][2]);
            let v2: Vec3 = Vec3::new(vertices[i + 2][0], vertices[i + 2][1], vertices[i + 2][2]);

            bounds.grow(v0);
            bounds.grow(v1);
            bounds.grow(v2);

            let e1: Vec3 = v1 - v0;
            let e2: Vec3 = v2 - v0;

            let tex0: Vec2 = Vec2::from(uvs[i]);
            let tex1: Vec2 = Vec2::from(uvs[i + 1]);
            let tex2: Vec2 = Vec2::from(uvs[i + 2]);

            let uv1: Vec2 = tex1 - tex0;
            let uv2: Vec2 = tex2 - tex0;

            let n = e1.cross(e2).normalize();

            let (t, b) = if uv1.dot(uv1) == 0.0 || uv2.dot(uv2) == 0.0 {
                let tangent: Vec3 = e1.normalize();
                let bitangent: Vec3 = n.cross(tangent).normalize();
                (tangent.extend(0.0), bitangent)
            } else {
                let r = 1.0 / (uv1.x * uv2.y - uv1.y * uv2.x);
                let tangent: Vec3 = (e1 * uv2.y - e2 * uv1.y) * r;
                let bitangent: Vec3 = (e1 * uv2.x - e2 * uv1.x) * r;
                (tangent.extend(0.0), bitangent)
            };

            for i in 0..3 {
                tangents[i + 0][i] += t[i];
                tangents[i + 1][i] += t[i];
                tangents[i + 2][i] += t[i];

                bitangents[i + 0][i] += b[i];
                bitangents[i + 1][i] += b[i];
                bitangents[i + 2][i] += b[i];
            }
        }

        let bounds = bounds;

        for i in 0..vertices.len() {
            let n: Vec3 = Vec3::from(normals[i]);
            let tangent = Vec4::from(tangents[i]).truncate().normalize();
            let bitangent = Vec3::from(bitangents[i]).normalize();

            let t: Vec3 = (tangent - (n * n.dot(tangent))).normalize();
            let c: Vec3 = n.cross(t);

            let w = c.dot(bitangent).signum();
            let t = tangent.normalize().extend(w);
            for j in 0..4 {
                tangents[i][j] = t[j];
            }
        }

        let mut last_id = material_ids[0];
        let mut start = 0;
        let mut range = 0;
        let mut meshes: Vec<VertexMesh> = Vec::new();
        let mut v_bounds = AABB::new();

        for i in 0..material_ids.len() {
            range += 1;
            v_bounds.grow([vertices[i][0], vertices[i][1], vertices[i][2]]);

            if last_id != material_ids[i] {
                meshes.push(VertexMesh {
                    first: start,
                    last: (start + range),
                    mat_id: last_id as i32,
                    bounds: v_bounds.clone(),
                });

                v_bounds = AABB::new();
                last_id = material_ids[i];
                start = i as u32;
                range = 1;
            }
        }

        if meshes.is_empty() {
            // There only is 1 mesh available
            meshes.push(VertexMesh {
                first: 0,
                last: vertices.len() as u32,
                mat_id: material_ids[0] as i32,
                bounds: bounds.clone(),
            });
        } else if (start + range) != (material_ids.len() as u32 - 1) {
            // Add last mesh to list
            meshes.push(VertexMesh {
                first: start,
                last: (start + range),
                mat_id: last_id as i32,
                bounds: v_bounds,
            })
        }

        Self {
            vertices,
            normals,
            uvs,
            tangents,
            material_ids: Vec::from(material_ids),
            materials,
            meshes,
            skeleton,
            bounds,
            name: name.unwrap_or(String::new()),
        }
    }

    pub fn scale(&self, scaling: f32) -> Self {
        let mut new_self = self.clone();

        let scaling = Mat4::from_scale(Vec3::splat(scaling));
        new_self.vertices.par_iter_mut().for_each(|t| {
            let mut v = Vec4::from(*t);
            v[3] = 1.0;
            let v = scaling * v;
            *t = [v.x, v.y, v.z, 1.0];
        });

        new_self
    }

    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    pub fn empty() -> Self {
        Self {
            vertices: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            tangents: Vec::new(),
            material_ids: Vec::new(),
            materials: None,
            meshes: Vec::new(),
            skeleton: None,
            bounds: AABB::new(),
            name: String::new(),
        }
    }

    /// Number of bytes required to store vertices, normals, uvs and tangents
    pub fn buffer_size(&self) -> usize {
        let f32_size = std::mem::size_of::<f32>();
        self.vertices.len() * 4
            + self.normals.len() * 3
            + self.uvs.len() * 2
            + self.tangents.len() * 4 * f32_size
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.vertices.as_ptr() as *const u8, self.buffer_size())
        }
    }
}

impl Bounds for MeshDescriptor {
    fn bounds(&self) -> AABB {
        self.bounds.clone()
    }
}
