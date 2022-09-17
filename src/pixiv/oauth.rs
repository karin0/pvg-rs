use crate::pixiv::error::{Error, Result};
use crate::pixiv::model::{Response, User};
use log::info;
use md5::{digest::Update, Digest, Md5};
use once_cell::sync::Lazy;
use reqwest::RequestBuilder;
use serde::Deserialize;
use serde_json::from_str;
use time::format_description;
use time::format_description::FormatItem;

const CLIENT_ID: &str = "MOBrBDS8blbauoSck0ZfDbtuzpyT";
const CLIENT_SECRET: &str = "lsACyCD94FhDUtGTXi3QzcFE2uU1hqtDaKeqrdwj";
const HASH_SECRET: &str = "28c1fdd170a5204386cb1313c7077b34f83e4aaf4aa829ce78c231e05b0bae2c";

static TIME_FORMAT: Lazy<Vec<FormatItem>> = Lazy::new(|| {
    format_description::parse("[year]-[month]-[day]T[hour]:[minute]:[second]+00:00").unwrap()
});

// Caller must set the user-agent header
pub async fn auth(req: RequestBuilder, refresh_token: &str) -> Result<Response> {
    let local_time = time::OffsetDateTime::now_utc()
        .format(&TIME_FORMAT)
        .unwrap();
    let hash = Md5::new().chain(&local_time).chain(HASH_SECRET).finalize();
    let hash = format!("{:x}", hash);
    let res = req
        .header("X-Client-Time", local_time)
        .header("X-Client-Hash", hash)
        .form(&[
            ("get_secure_url", "1"),
            ("client_id", CLIENT_ID),
            ("client_secret", CLIENT_SECRET),
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
        ])
        .send()
        .await?;
    let code = res.status().as_u16();
    let text = res.text().await?;
    info!("auth: {} {}", code, text);
    match code {
        200 | 301 | 302 => Ok(from_str(&text)?),
        _ => Err(Error::Auth(code, text)),
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct AuthResponse {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_in: u32,
    pub user: User,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AuthSuccess {
    pub response: AuthResponse,
}
