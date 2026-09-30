use crate::Error;
use crate::error::Result;
use log::error;
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::{Client, Response, header};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct DownloadClient {
    client: Client,
}

impl Default for DownloadClient {
    fn default() -> Self {
        Self::new()
    }
}

pub type DownloadResponse = Response;

impl DownloadClient {
    pub fn new() -> Self {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::ACCEPT,
            HeaderValue::from_static("text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9")
        );
        headers.insert(
            header::REFERER,
            HeaderValue::from_static("https://www.pixiv.net/"),
        );
        Self {
            client: Client::builder()
                .connect_timeout(Duration::from_secs(30))
                .timeout(Duration::from_mins(1))
                .default_headers(headers)
                .user_agent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36")
                .build().unwrap()
        }
    }

    pub async fn download(&self, url: &str) -> Result<Response> {
        let r = self.client.get(url).send().await?;
        let st = r.status();
        // info!("{} from {}", st, url);
        if st.is_success() || st.is_redirection() {
            Ok(r)
        } else {
            error!("download: {st:?} from {url}");
            Err(Error::Pixiv(st.as_u16(), r.text().await?))
        }
    }
}

/// The name an original keeps once an artist's re-upload reuses its URL
/// basename: the digits of the URL before the basename, which pixiv's date
/// path changes with each upload, go before the extension. So
/// `.../img/2024/01/02/03/04/05/12345678_p0.png` keeps
/// `12345678_p0@20240102030405.png`.
#[must_use]
pub fn versioned_name(url: &str) -> String {
    let (dir, name) = url.rsplit_once('/').unwrap_or(("", url));
    let digits: String = dir.chars().filter(char::is_ascii_digit).collect();
    match name.rsplit_once('.') {
        Some((stem, ext)) => format!("{stem}@{digits}.{ext}"),
        None => format!("{name}@{digits}"),
    }
}

#[cfg(test)]
mod tests {
    use super::versioned_name;

    #[test]
    fn keeps_the_digits_before_the_basename() {
        assert_eq!(
            versioned_name(
                "https://i.pximg.net/img-original/img/2024/01/02/03/04/05/12345678_p0.png"
            ),
            "12345678_p0@20240102030405.png"
        );
    }
}
