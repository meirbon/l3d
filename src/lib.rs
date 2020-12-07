mod load;
mod mat;

use std::{collections::HashMap, error::Error, fmt::Display, path::PathBuf, write};

use load::{Loader, MeshDescriptor, SceneDescriptor};

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
    FileDoesNotExist(PathBuf),
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
                _ => String::new(),
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

    pub fn with_default(mut self) -> Self {
        // TODO
        self
    }

    pub fn with_loader<T: Loader + Sized + Clone + 'static>(mut self, loader: T) -> Self {
        let file_names = loader.file_extensions();
        for f in file_names {
            let f = String::from(f);
            let loader: Box<dyn Loader> = Box::new(loader.clone());
            self.loaders.insert(f, loader);
        }

        self
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
