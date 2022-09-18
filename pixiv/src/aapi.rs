use crate::client::{ApiState, Client};
use crate::endpoint::Endpoint;
use crate::error::{Error, Result};
use log::info;
use reqwest::{Method, RequestBuilder};
use serde::de::DeserializeOwned;
use strum_macros::IntoStaticStr;
use url::Url;

async fn fin<'de, T: DeserializeOwned>(req: RequestBuilder) -> Result<T> {
    info!("sending: {:?}", req);
    let r = req.send().await?;
    let st = r.status();
    info!("got status: {:?}", st);
    if st.is_success() || st.is_redirection() {
        Ok(r.json().await?)
    } else {
        Err(Error::Pixiv(st.as_u16(), r.text().await?))
    }
}

#[derive(IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum BookmarkRestrict {
    Public,
    Private,
}

impl<S: ApiState> Client<S> {
    fn app(&self, endpoint: &impl Endpoint) -> RequestBuilder {
        self.call(endpoint).header("host", "app-api.pixiv.net")
    }

    pub async fn next<T: DeserializeOwned>(&self, next_url: &str) -> Result<T> {
        fin(self.app(&(Method::GET, Url::parse(next_url)?))).await
    }

    pub async fn user_bookmarks_illust<T: DeserializeOwned>(
        &self,
        user_id: &str,
        restrict: BookmarkRestrict,
    ) -> Result<T> {
        fin(self.app(&self.api.user_bookmarks_illust).query(&[
            ("user_id", user_id),
            ("restrict", restrict.into()),
            ("filter", "for_ios"),
        ]))
        .await
    }
}
