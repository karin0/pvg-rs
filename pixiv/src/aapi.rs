use crate::client::{ApiState, Client};
use crate::endpoint::Endpoint;
use crate::error::{Error, Result};
use log::{debug, error};
use reqwest::{Method, RequestBuilder};
use serde::de::DeserializeOwned;
use strum_macros::IntoStaticStr;
use url::Url;

async fn finalize<T: DeserializeOwned>(req: RequestBuilder) -> Result<T> {
    let r = req.send().await?;
    let st = r.status();
    if st.is_success() || st.is_redirection() {
        debug!("{} from {}", st, r.url());
        Ok(r.json().await?)
    } else {
        error!("{} from {}", st, r.url());
        Err(Error::Pixiv(st.as_u16(), r.text().await?))
    }
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
}
