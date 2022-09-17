use crate::pixiv::endpoint::Endpoint;
use crate::pixiv::error::{Error, Result};
use crate::pixiv::model::{from_response, AuthResponse, AuthSuccess, Response};
use log::info;
use md5::{digest::Update, Digest, Md5};
use once_cell::sync::Lazy;
use reqwest::{Client, RequestBuilder, Url};
use serde::Deserializer;
use serde_json::{from_str, Value};
use time::format_description;
use time::format_description::FormatItem;

const CLIENT_ID: &str = "MOBrBDS8blbauoSck0ZfDbtuzpyT";
const CLIENT_SECRET: &str = "lsACyCD94FhDUtGTXi3QzcFE2uU1hqtDaKeqrdwj";
const HASH_SECRET: &str = "28c1fdd170a5204386cb1313c7077b34f83e4aaf4aa829ce78c231e05b0bae2c";

pub struct AuthState {
    token: AuthResponse,
    time: Instant,
}

#[derive(Default)]
pub struct Session {
    pub user_agent: String,
    pub token: Option<AuthResponse>,
    http: Client,
}

static TIME_FORMAT: Lazy<Vec<FormatItem>> = Lazy::new(|| {
    format_description::parse("[year]-[month]-[day]T[hour]:[minute]:[second]+00:00").unwrap()
});

impl Session {
    pub(crate) fn new() -> Self {
        Self {
            user_agent: "PixivAndroidApp/5.0.234 (Android 11; Pixel 5)".into(),
            ..Default::default()
        }
    }

    pub(crate) fn call(&self, endpoint: &impl Endpoint) -> RequestBuilder {
        endpoint.request(&self.http)
    }

    async fn do_auth(&self, endpoint: &impl Endpoint, refresh_token: &str) -> Result<Response> {
        let local_time = time::OffsetDateTime::now_utc()
            .format(&TIME_FORMAT)
            .unwrap();
        let hash = Md5::new().chain(&local_time).chain(HASH_SECRET).finalize();
        let hash = format!("{:x}", hash);
        let res = self
            .call(endpoint)
            .header("User-Agent", &self.user_agent)
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

    pub(crate) async fn auth(
        &mut self,
        endpoint: &impl Endpoint,
        refresh_token: &str,
    ) -> Result<Response> {
        let resp = self.do_auth(endpoint, refresh_token).await?;
        let r: AuthSuccess = from_response(resp.clone())?;
        self.token = Some(r.response);
        Ok(resp)
    }

    async fn reauth(&mut self, endpoint: &impl Endpoint) -> Result<Response> {
        let refresh_token = &self
            .token
            .as_ref()
            .ok_or_else(Error::Unauthed)?
            .refresh_token;
        let resp = self.do_auth(endpoint, refresh_token).await?;
        let r: AuthSuccess = from_response(resp.clone())?;
        self.token = Some(r.response);
        Ok(resp)
    }
}
