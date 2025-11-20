#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

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
