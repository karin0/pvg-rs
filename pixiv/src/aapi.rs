use crate::client::{ApiState, Client};
use crate::endpoint::Endpoint;
use crate::error::{Error, Result};
use crate::model::IllustId;
use log::{debug, error};
use reqwest::{Method, RequestBuilder, Response};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use strum_macros::IntoStaticStr;
use url::Url;

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

/// pixiv states a failure in one of three message slots and leaves the rest
/// empty, escaping any non-ASCII text.
fn describe(body: String) -> String {
    match serde_json::from_str::<ErrorEnvelope>(&body) {
        Ok(env) => {
            let e = env.error;
            [e.user_message, e.message, e.reason]
                .into_iter()
                .find(|s| !s.is_empty())
                .unwrap_or(body)
        }
        Err(_) => body,
    }
}

async fn send(req: RequestBuilder) -> Result<Response> {
    let r = req.send().await?;
    let st = r.status();
    if st.is_success() || st.is_redirection() {
        debug!("{} from {}", st, r.url());
        Ok(r)
    } else {
        error!("{} from {}", st, r.url());
        Err(Error::Pixiv(st.as_u16(), describe(r.text().await?)))
    }
}

async fn finalize<T: DeserializeOwned>(req: RequestBuilder) -> Result<T> {
    Ok(send(req).await?.json().await?)
}

#[derive(Copy, Clone, Debug, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum Restrict {
    Public,
    Private,
}

#[deprecated]
pub type BookmarkRestrict = Restrict;

impl<S: ApiState> Client<S> {
    fn app(&self, endpoint: &impl Endpoint) -> RequestBuilder {
        self.call(endpoint).header("host", "app-api.pixiv.net")
    }

    pub async fn call_url<T: DeserializeOwned>(&self, url: &str) -> Result<T> {
        finalize(self.app(&(Method::GET, Url::parse(url)?))).await
    }

    pub async fn user_bookmarks_illust<T: DeserializeOwned>(
        &self,
        user_id: &str,
        restrict: Restrict,
    ) -> Result<T> {
        finalize(self.app(&self.api.user_bookmarks_illust).query(&[
            ("user_id", user_id),
            ("restrict", restrict.into()),
            ("filter", "for_ios"),
        ]))
        .await
    }

    pub async fn illust_follow<T: DeserializeOwned>(&self, restrict: Restrict) -> Result<T> {
        finalize(
            self.app(&self.api.illust_follow)
                .query(&[("restrict", Into::<&str>::into(restrict))]),
        )
        .await
    }

    pub async fn illust_bookmark_add(&self, id: IllustId, restrict: Restrict) -> Result<()> {
        let id = id.to_string();
        send(
            self.app(&self.api.illust_bookmark_add)
                .form(&[("illust_id", id.as_str()), ("restrict", restrict.into())]),
        )
        .await?;
        Ok(())
    }
}
