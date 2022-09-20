use aho_corasick::AhoCorasickBuilder;
use anyhow::{Context, Result};
use pixiv::model as api;
use pixiv::IllustId;
use serde::de::value::MapDeserializer;
use serde::Deserialize;
use serde_json::{from_str, Map, Value};
use std::collections::{HashMap, HashSet};
use std::num::NonZeroU32;

#[derive(Debug, Clone, Copy)]
pub struct Dimensions(pub NonZeroU32, pub NonZeroU32);

impl Dimensions {
    pub fn new(width: u32, height: u32) -> Result<Self> {
        Ok(Self(
            width.try_into().context("width is zero")?,
            height.try_into().context("height is zero")?,
        ))
    }
}

#[derive(Debug, Clone)]
pub struct Source {
    pub url: String,
    p: u32,
}

impl TryFrom<String> for Source {
    type Error = anyhow::Error;

    fn try_from(url: String) -> Result<Self> {
        let p = url.rfind('/').context("bad image url")?;
        Ok(Self {
            url,
            p: p as u32 + 1,
        })
    }
}

impl Source {
    pub fn filename(&self) -> &str {
        &self.url[self.p as usize..]
    }
}

#[derive(Debug, Clone)]
pub struct Page {
    pub source: Source,
    pub dimensions: Option<Dimensions>,
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
    ids: Vec<IllustId>, // TODO: store pointers to speed up?
    staged: Vec<IllustId>,
}

impl Page {
    pub fn new(url: String, dimensions: Option<Dimensions>) -> Result<Self> {
        Ok(Self {
            source: url.try_into()?,
            dimensions,
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
                Some(Dimensions::new(data.width, data.height)?),
            )?]
        } else {
            let mut it = data.meta_pages.iter();
            let first = it.next().context("no pages")?;
            let mut vec = Vec::with_capacity(data.page_count as usize);
            vec.push(Page::new(
                first.image_urls.original.clone(),
                Some(Dimensions::new(data.width, data.height)?),
            )?);
            for p in it {
                vec.push(Page::new(p.image_urls.original.clone(), None)?);
            }
            vec
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

pub type DimCache = Vec<(IllustId, Vec<u32>)>;

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

    pub fn load_dims_cache(&mut self, cache: DimCache) -> Result<()> {
        let mut cnt = 0;
        for (iid, a) in cache.iter() {
            if let Some(i) = self.map.get_mut(iid) {
                cnt += 1;
                let pc = i.data.page_count as usize;
                if (pc as usize - 1) * 2 == a.len() {
                    for p in 1..pc {
                        let w = a[(p - 1) << 1];
                        let h = a[(p - 1) << 1 | 1];
                        i.pages[p].dimensions = Some(Dimensions::new(w, h)?);
                    }
                } else {
                    warn!(
                        "inconsistent cache size: {} has {}, cache has {}",
                        iid,
                        pc,
                        a.len()
                    );
                }
            } else {
                warn!("no such illust: {}", iid);
            }
        }
        info!("loaded {} dimensions from cache", cnt);
        Ok(())
    }

    pub fn dump_dims_cache(&self) -> Vec<(IllustId, Vec<u32>)> {
        let mut res = vec![];
        for i in self.iter() {
            if i.data.page_count > 1 {
                let mut a = Vec::with_capacity((i.data.page_count as usize - 1) * 2);
                let mut any = false;
                for p in i.pages.iter().skip(1) {
                    if let Some(d) = p.dimensions {
                        a.push(d.0.get());
                        a.push(d.1.get());
                        any = true;
                    } else {
                        a.push(0);
                        a.push(0);
                    }
                }
                if any {
                    res.push((i.data.id, a));
                }
            }
        }
        res
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &Illust> {
        self.ids.iter().map(move |id| &self.map[id])
    }

    pub fn stage(&mut self, illust: Map<String, Value>) -> Result<bool> {
        let illust = Illust::new(illust)?;
        let id = illust.data.id;
        if !illust.data.visible {
            let has = self.map.contains_key(&id);
            if has {
                warn!(
                    "{} ({} - {}) is deleted from upstream, which has luckily been indexed.",
                    id, illust.data.title, illust.data.user.name
                );
                return Ok(false);
            }
            warn!(
                "{} ({} - {}) is deleted from upstream, which sadly hasn't been indexed.",
                id, illust.data.title, illust.data.user.name
            );
            let r = self.map.insert(id, illust);
            assert!(r.is_none());
        } else if self.map.insert(illust.data.id, illust).is_some() {
            // TODO: how to know if this illust is updated?
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

    pub fn select(&self, filters: &[String]) -> Box<dyn DoubleEndedIterator<Item = &Illust> + '_> {
        if filters.is_empty() {
            return Box::new(self.iter());
        }
        let ac = AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .build(filters);
        let mut s = HashSet::new();
        let n = filters.len();
        Box::new(self.iter().filter(move |illust| {
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
