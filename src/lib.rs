pub mod load;
pub mod mat;
pub mod prelude {
    pub use crate::load::*;
    pub use crate::mat::*;
    pub use crate::*;
}

use crate::load::LoadOptions;
use load::{Loader, MeshDescriptor, SceneDescriptor};
use std::{collections::HashMap, error::Error, fmt::Display, path::PathBuf, write};

#[derive(Debug)]
pub enum LoadResult {
    /// Reference to single mesh
    Mesh(MeshDescriptor),
    /// Indices of root nodes of scene
    Scene(SceneDescriptor),
    None(LoadError),
}

impl LoadResult {
    pub fn mesh(&self) -> Option<&MeshDescriptor> {
        match self {
            Self::Mesh(m) => Some(m),
            _ => None,
        }
    }

    pub fn scene(&self) -> Option<&SceneDescriptor> {
        match self {
            Self::Scene(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum LoadError {
    NoFileExtension,
    CouldNotParseSource,
    CouldNotParseExtension,
    UnsupportedExtension(String),
    FileDoesNotExist(PathBuf),
    InvalidFile(PathBuf),
    TextureDoesNotExist(PathBuf),
    Error(Box<dyn Error>),
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
                LoadError::CouldNotParseSource => String::from("Could not parse given source"),
                LoadError::CouldNotParseExtension => String::from("Could not parse file extension"),
                LoadError::UnsupportedExtension(ext) =>
                    format!("Unsupported file extension: {}", ext),
                LoadError::Error(e) => format!("{}", e),
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
            self.loaders
                .insert(f, Box::new(load::gltf::GltfLoader::default()));
        }
        self
    }

    /// Adds a loader to this instance
    pub fn with_loader<T: Loader + Sized + Clone + 'static>(mut self, loader: T) -> Self {
        let file_names = loader.file_extensions();
        for f in file_names {
            let loader: Box<dyn Loader> = Box::new(loader.clone());
            self.loaders.insert(f, loader);
        }

        self
    }

    /// Loads file given in LoadOptions
    pub fn load(&self, options: LoadOptions) -> LoadResult {
        let extension = match &options.source {
            load::LoadSource::Path(path) => match path.extension() {
                Some(e) => match e.to_str() {
                    Some(e) => e,
                    None => return LoadResult::None(LoadError::NoFileExtension),
                },
                None => return LoadResult::None(LoadError::NoFileExtension),
            },
            load::LoadSource::String { extension, .. } => extension,
        }
        .to_string();

        if let Some(loader) = self.loaders.get(&extension) {
            loader.load(options)
        } else {
            LoadResult::None(LoadError::UnsupportedExtension(extension))
        }
    }
}

impl Default for LoadInstance {
    fn default() -> Self {
        Self::new().with_default()
    }
}
