#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]

#[macro_use]
extern crate log;

pub mod config;
pub mod core;
pub mod disk_lru;
pub mod download;
pub mod hook;
pub mod illust;
pub mod model;
pub mod store;
pub mod upscale;
pub mod util;

#[cfg(feature = "search")]
pub mod search;
