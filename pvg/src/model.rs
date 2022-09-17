use aho_corasick::AhoCorasickBuilder;
use anyhow::{Context, Result};
use either::Either;
use pixiv::model as api;
use pixiv::IllustId;
use serde::de::value::MapDeserializer;
use serde::Deserialize;
use serde_json::{from_str, Map, Value};
use std::collections::{HashMap, HashSet};
use std::io::ErrorKind;
use std::path::Path;
use tokio::fs;

#[derive(Debug)]
pub struct Page {
    pub url: String,
    pub filename: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug)]
pub struct Illust {
    pub(crate) data: api::Illust,
    raw_data: Map<String, Value>,
    pub(crate) pages: Vec<Page>,
    intro: String,
}

#[derive(Debug)]
pub struct IllustIndex {
    pub map: HashMap<IllustId, Illust>,
    pub ids: Vec<IllustId>, // TODO: store pointers to speed up?
}

impl Page {
    pub fn new(url: String, width: u32, height: u32) -> Result<Self> {
        let p = url.rfind('/').context("bad image url")?;
        let filename = url[p + 1..].to_string();
        Ok(Self {
            url,
            filename,
            width,
            height,
        })
    }
}

fn make_intro(data: &api::Illust) -> String {
    let mut s = format!("{}\\{}", data.title, data.user.name);
    for t in &data.tags {
        s.push('\\');
        s.push_str(&t.name);
    }
    s.push_str(if data.sanity_level >= 6 {
        "\\$nsfw"
    } else {
        "\\$sfw"
    });
    s
}

impl Illust {
    fn new(raw_data: Map<String, Value>) -> Result<Self> {
        let data = MapDeserializer::new(raw_data.clone().into_iter());
        let data = api::Illust::deserialize(data)?;
        let pages = if data.page_count == 1 {
            vec![Page::new(
                data.meta_single_page
                    .original_image_url
                    .as_ref()
                    .context("Bad image url")?
                    .clone(),
                data.width,
                data.height,
            )?]
        } else {
            data.meta_pages
                .iter()
                .zip(
                    // TODO: this requires old pvg!
                    raw_data
                        .get("sizes")
                        .context("no size!")?
                        .as_array()
                        .unwrap()
                        .iter(),
                )
                .filter_map(|(p, v)| -> Option<Result<Page>> {
                    let v: &Value = v;
                    let v = v.as_array()?;
                    if let Some(w) = v[0].as_u64() {
                        if let Some(h) = v[1].as_u64() {
                            return Some(Page::new(
                                p.image_urls.original.clone(),
                                w as u32,
                                h as u32,
                            ));
                        }
                    }
                    warn!("bad page: iid = {}, size = {:?}", data.id, v);
                    None
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        let intro = make_intro(&data);
        Ok(Self {
            data,
            raw_data,
            pages,
            intro,
        })
    }
}

impl IllustIndex {
    pub async fn new(file: &Path) -> Result<Self> {
        let s = fs::read_to_string(file).await;
        match s {
            Ok(s) => {
                info!("read {} bytes", s.len());
                let illusts: Vec<Map<String, Value>> = from_str(&s)?;
                let mut ids = Vec::with_capacity(illusts.len());
                info!("parsed {} objects", illusts.len());

                let map = HashMap::from_iter(illusts.into_iter().map(|data| {
                    // FIXME: do not unwrap
                    let o = Illust::new(data).unwrap();
                    ids.push(o.data.id);
                    (o.data.id, o)
                }));

                Ok(Self { map, ids })
            }
            Err(e) => {
                if e.kind() == ErrorKind::NotFound {
                    Ok(Self {
                        map: HashMap::new(),
                        ids: Vec::new(),
                    })
                } else {
                    Err(e.into())
                }
            }
        }
    }

    fn iter(&self) -> impl Iterator<Item = &Illust> {
        self.ids.iter().map(move |id| &self.map[id])
    }

    pub fn select(&self, filters: &[String]) -> impl Iterator<Item = &Illust> {
        if filters.is_empty() {
            return Either::Left(self.iter());
        }
        let ac = AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .build(filters);
        let mut s = HashSet::new();
        let n = filters.len();
        Either::Right(self.iter().filter(move |illust| {
            s.clear();
            for mat in ac.find_overlapping_iter(&illust.intro) {
                s.insert(mat.pattern());
                if s.len() == n {
                    return true;
                }
            }
            false
        }))
    }
}
