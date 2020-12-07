pub mod load;
pub mod mat;

use std::{collections::HashMap, error::Error, fmt::Display, path::PathBuf, write};
use load::{Loader, MeshDescriptor, SceneDescriptor};
use crate::load::LoadOptions;

#[derive(Debug, Clone)]
pub enum LoadResult {
    /// Reference to single mesh
    Mesh(MeshDescriptor),
    /// Indices of root nodes of scene
    Scene(SceneDescriptor),
    None(LoadError),
}

#[derive(Debug, Clone)]
pub enum LoadError {
    NoFileExtension,
    UnsupportedExtension(String),
    FileDoesNotExist(PathBuf),
    InvalidFile(PathBuf),
    TextureDoesNotExist(PathBuf),
}

impl Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                LoadError::FileDoesNotExist(file) =>
                    format!("File does not exist: {}", file.display()),
                LoadError::TextureDoesNotExist(file) =>
                    format!("Texture does not exist: {}", file.display()),
                LoadError::InvalidFile(file) =>
                    format!("File might be corrupted: {}", file.display()),
                LoadError::NoFileExtension => String::from("No file extension"),
                LoadError::UnsupportedExtension(ext) => format!("Unsupported file extension: {}", ext)
            }
        )
    }
}

impl Error for LoadError {}

pub struct LoadInstance {
    loaders: HashMap<String, Box<dyn Loader>>,
}

impl LoadInstance {
    pub fn new() -> Self {
        Self {
            loaders: HashMap::new(),
        }
    }

    /// Loads default loaders included with this library
    pub fn with_default(mut self) -> Self {
        let obj_loader = load::obj::ObjLoader::default();
        for f in obj_loader.file_extensions() {
            self.loaders
                .insert(f, Box::new(load::obj::ObjLoader::default()));
        }

        let gltf_loader = load::gltf::GltfLoader::default();
        for f in gltf_loader.file_extensions() {
            self.loaders.insert(f, Box::new(load::gltf::GltfLoader::default()));
        }
        self
    }

    /// Adds a loader to this instance
    pub fn with_loader<T: Loader + Sized + Clone + 'static>(mut self, loader: T) -> Self {
        let file_names = loader.file_extensions();
        for f in file_names {
            let f = String::from(f);
            let loader: Box<dyn Loader> = Box::new(loader.clone());
            self.loaders.insert(f, loader);
        }

        self
    }

    /// Loads file given in LoadOptions
    pub fn load(&self, options: LoadOptions) -> LoadResult {
        let ext = match options.path.extension() {
            Some(e) => e,
            None => return LoadResult::None(LoadError::NoFileExtension),
        };

        let ext = ext.to_str().unwrap().to_string();
        if let Some(loader) = self.loaders.get(&ext) {
            loader.load(options)
        } else {
            return LoadResult::None(LoadError::UnsupportedExtension(ext));
        }
    }
}