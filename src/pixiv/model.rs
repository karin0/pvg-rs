use serde::Deserialize;

pub type IllustId = u32;
pub type PageNum = u32;

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

#[derive(Debug, Deserialize)]
pub struct User {
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
