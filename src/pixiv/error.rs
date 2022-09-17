use crate::pixiv::model::Response;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("reqwest error")]
    Request(#[from] reqwest::Error),
    #[error("serde_json error")]
    Json(#[from] serde_json::Error),
    #[error("api failed")]
    Pixiv(Response),
    #[error("auth failed")]
    Auth(u16, String),
}

pub type Result<T> = std::result::Result<T, Error>;
