#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("reqwest error")]
    Request(#[from] reqwest::Error),
    #[error("json deserialization failed")]
    Json(#[from] serde_json::Error),
    #[error("api error")]
    Pixiv(u16, String),
    #[error("url parse failed")]
    Url(#[from] url::ParseError),
}

pub type Result<T> = std::result::Result<T, Error>;
