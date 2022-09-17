mod aapi;
mod endpoint;
mod error;
pub mod model;
mod session;

pub use model::IllustId;
pub use model::PageNum;

pub use aapi::AppApi;
pub use error::{Error, Result};
