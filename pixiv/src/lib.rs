pub mod aapi;
pub mod client;
pub mod download;
mod endpoint;
mod error;
pub mod model;
mod oauth;

pub use error::{Error, Result};
pub use model::{IllustId, PageNum};
pub use reqwest;
