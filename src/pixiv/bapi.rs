use reqwest::{Client, Request, Url};
use time::format_description;
use time::format_description::FormatItem;
use once_cell::sync::Lazy;
use md5::{Md5, Digest, digest::Update};
use thiserror::Error;

#[derive(Default)]
struct Session<'a> {
    client_id: &'a str,
    client_secret: &'a str,
    hash_secret: &'a str,
    user_agent: String,
    access_token: Option<String>,
    refresh_token: Option<String>,
    http: Client
}

static TIME_FORMAT: Lazy<Vec<FormatItem>> = Lazy::new(|| {
    format_description::parse(
        "[year]-[month]-[day]T[hour]:[minute]:[second]+00:00",
    ).unwrap()
});

impl<'a> Session<'a> {
    fn new() -> Self {
        Self {
            client_id: "MOBrBDS8blbauoSck0ZfDbtuzpyT",
            client_secret: "lsACyCD94FhDUtGTXi3QzcFE2uU1hqtDaKeqrdwj",
            hash_secret: "28c1fdd170a5204386cb1313c7077b34f83e4aaf4aa829ce78c231e05b0bae2c",
            user_agent: "PixivAndroidApp/5.0.234 (Android 11; Pixel 5)".into(),

            ..Default::default()
        }
    }

    async fn login(&mut self, url: &str, username: &str, refresh_token: &str) {
        let local_time = time::OffsetDateTime::now_utc().format(&TIME_FORMAT).unwrap();
        let hash = Md5::new()
            .chain(&local_time)
            .chain(&self.hash_secret)
            .finalize();
        let hash = format!("{:x}", hash);
        let req = self.http.post(url)
            .header("User-Agent", &self.user_agent)
            .header("X-Client-Time", local_time)
            .header("X-Client-Hash", hash)
            .form(&[
                ("get_secure_url", "1"),
                ("client_id", self.client_id),
                ("client_secret", self.client_secret),
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token),
            ])
            .send()
            .await;
    }
}
