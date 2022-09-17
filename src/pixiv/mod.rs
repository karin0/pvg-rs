pub mod aapi;
pub mod client;
mod endpoint;
mod error;
pub mod model;
mod oauth;

pub use model::IllustId;
pub use model::PageNum;

pub use error::{Error, Result};
