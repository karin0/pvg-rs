use serde::de::value::MapDeserializer;
use serde::Deserialize;
use serde_aux::field_attributes::deserialize_number_from_string;
use serde_json::{Map, Value};

pub type IllustId = u32;
pub type PageNum = u32;

pub type Response = Map<String, Value>;

pub fn from_response<'de, T: Deserialize<'de>>(resp: Response) -> Result<T, serde_json::Error> {
    T::deserialize(MapDeserializer::new(resp.into_iter()))
}

#[derive(Debug, Deserialize)]
pub struct ImageUrls {
    pub original: String,
}

#[derive(Debug, Deserialize)]
pub struct MetaPage {
    pub image_urls: ImageUrls,
}

#[derive(Debug, Deserialize)]
pub struct MetaSinglePage {
    pub original_image_url: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Tag {
    pub name: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct User {
    // id is typed number in user_bookmarks_illust, but string in auth.
    #[serde(deserialize_with = "deserialize_number_from_string")]
    pub id: u32,
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct Illust {
    pub id: IllustId,
    pub title: String,
    pub user: User,
    pub width: u32,
    pub height: u32,
    pub meta_single_page: MetaSinglePage,
    pub page_count: u32,
    pub meta_pages: Vec<MetaPage>,
    pub tags: Vec<Tag>,
    pub sanity_level: u32,
}

#[derive(Deserialize, Debug)]
pub struct UserBookmarksIllust {
    pub illusts: Vec<Illust>,
    pub next_url: Option<String>,
}
