use crate::pixiv::client::{ApiState, Client};
use crate::pixiv::endpoint::Endpoint;
use crate::pixiv::error::Result;
use crate::pixiv::model::{Illust, Response};
use reqwest::RequestBuilder;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use typed_builder::TypedBuilder;

async fn fin<'de, T: DeserializeOwned>(req: RequestBuilder) -> Result<T> {
    Ok(req.send().await?.json().await?)
}

#[derive(Debug, TypedBuilder)]
pub struct UserBookmarksIllust<'a> {
    #[builder(setter(into))]
    pub user_id: &'a str,
    #[builder(default = "public")]
    pub restrict: &'a str,
    #[builder(default = "for_ios")]
    pub filter: &'a str,
}

#[derive(Deserialize, Debug)]
pub struct UserBookmarksIllustResponse {
    pub illusts: Vec<Illust>,
    pub next_url: Option<String>,
}

impl<S: ApiState> Client<S> {
    fn app(&self, endpoint: &impl Endpoint) -> RequestBuilder {
        self.call(endpoint).header("host", "app-api.pixiv.net")
    }

    pub async fn user_bookmarks_illust(&self, i: UserBookmarksIllust<'_>) -> Result<Response> {
        fin(self.app(&self.api.user_bookmarks_illust).query(&[
            ("user_id", i.user_id),
            ("restrict", i.restrict),
            ("filter", i.filter),
        ]))
        .await
    }
}
