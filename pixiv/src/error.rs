use reqwest::Response;
use serde::Deserialize;
use serde::de::IgnoredAny;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("reqwest error")]
    Request(#[from] reqwest::Error),
    #[error("json deserialization failed")]
    Json(#[from] serde_json::Error),
    #[error("pixiv api {0}: {1}")]
    Pixiv(u16, String),
    #[error("url parse failed")]
    Url(#[from] url::ParseError),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Deserialize)]
struct ErrorEnvelope {
    error: ErrorMessages,
}

#[derive(Deserialize)]
struct ErrorMessages {
    #[serde(default)]
    user_message: String,
    #[serde(default)]
    message: String,
    #[serde(default)]
    reason: String,
}

/// pixiv states a failure in one of three message slots of the app API's
/// envelope and leaves the rest empty, escaping any non-ASCII text. Other JSON
/// is short and kept whole, while a page such as a Cloudflare challenge runs to
/// kilobytes of HTML and is reduced to its size.
fn describe(body: String) -> String {
    if let Ok(env) = serde_json::from_str::<ErrorEnvelope>(&body) {
        let e = env.error;
        if let Some(msg) = [e.user_message, e.message, e.reason]
            .into_iter()
            .find(|s| !s.is_empty())
        {
            return msg;
        }
    }
    if serde_json::from_str::<IgnoredAny>(&body).is_ok() {
        body
    } else {
        format!("{}-byte non-JSON body", body.len())
    }
}

/// The error of a response whose status is a failure.
pub(crate) async fn failure(r: Response) -> Error {
    let code = r.status().as_u16();
    match r.text().await {
        Ok(body) => Error::Pixiv(code, describe(body)),
        Err(e) => e.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::describe;

    #[test]
    fn takes_the_message_slot_of_the_envelope() {
        let body = r#"{"error":{"user_message":"","message":"Rate Limit","reason":""}}"#;
        assert_eq!(describe(body.to_owned()), "Rate Limit");
    }

    #[test]
    fn keeps_other_json_whole() {
        let body = r#"{"has_error":true,"errors":{"system":{"code":1508}}}"#;
        assert_eq!(describe(body.to_owned()), body);
    }

    #[test]
    fn reduces_a_page_to_its_size() {
        let body = "<!DOCTYPE html><html><title>Just a moment...</title></html>";
        assert_eq!(describe(body.to_owned()), "59-byte non-JSON body");
    }
}
