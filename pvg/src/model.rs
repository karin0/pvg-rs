use aho_corasick::AhoCorasickBuilder;
use anyhow::{Context, Result};
use either::Either;
use pixiv::model as api;
use pixiv::IllustId;
use serde::de::value::MapDeserializer;
use serde::Deserialize;
use serde_json::{from_str, Map, Value};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Page {
    pub url: String,
    pub filename: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone)]
pub struct Illust {
    pub(crate) data: api::Illust,
    raw_data: Map<String, Value>,
    pub(crate) pages: Vec<Page>,
    intro: String,
}

#[derive(Debug, Default, Clone)]
pub struct IllustIndex {
    pub map: HashMap<IllustId, Illust>,
    pub ids: Vec<IllustId>, // TODO: store pointers to speed up?
    pub staged: Vec<IllustId>,
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
    pub fn parse(s: String) -> serde_json::error::Result<Self> {
        let illusts: Vec<Map<String, Value>> = from_str(&s)?;
        let mut ids = Vec::with_capacity(illusts.len());
        info!("parsed {} objects", illusts.len());

        let map = HashMap::from_iter(illusts.into_iter().map(|data| {
            // FIXME: do not unwrap
            let o = Illust::new(data).unwrap();
            ids.push(o.data.id);
            (o.data.id, o)
        }));

        Ok(Self {
            map,
            ids,
            staged: vec![],
        })
    }

    pub fn dump(&self) -> serde_json::error::Result<Vec<u8>> {
        // FIXME: serde can't do async streaming for now.
        info!("collecting {} illlusts", self.len());
        let v = self.iter().map(|i| &i.raw_data).collect::<Vec<_>>();
        info!("dumping {} illlusts", v.len());
        serde_json::to_vec(&v)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    fn iter(&self) -> impl Iterator<Item = &Illust> {
        self.ids.iter().map(move |id| &self.map[id])
    }

    pub fn stage(&mut self, illust: Map<String, Value>) -> Result<bool> {
        let illust = Illust::new(illust)?;
        let id = illust.data.id;
        if self.map.insert(illust.data.id, illust).is_some() {
            return Ok(false);
        }
        self.staged.push(id);
        Ok(true)
    }

    pub fn commit(&mut self) -> usize {
        // self.staged.reverse();
        // self.ids.append(&mut self.staged);
        let delta = self.staged.len();
        let n = self.ids.len();
        self.ids.extend(self.staged.drain(..).rev());
        assert_eq!(self.ids.len(), n + delta);
        delta
    }

    pub fn rollback(&mut self) -> usize {
        let delta = self.staged.len();
        let n = self.map.len();
        for id in self.staged.drain(..) {
            self.map.remove(&id);
        }
        assert_eq!(self.map.len(), n - delta);
        delta
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
