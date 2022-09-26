use anyhow::{bail, Context, Result};
use itertools::Itertools;
use pixiv::model as api;
use pixiv::IllustId;
use serde::de::value::MapDeserializer;
use serde::Deserialize;
use serde_json::{from_str, Map, Value};
use std::collections::hash_map::RandomState;
use std::collections::{HashMap, HashSet};
use std::num::NonZeroU32;
use suffix::SuffixTable;

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
struct IllustStage {
    ids: Vec<IllustId>,
    dirty: bool,
}

#[derive(Debug, Clone)]
pub struct IllustIndex {
    pub map: HashMap<IllustId, Illust>,
    ids: Vec<IllustId>, // TODO: store pointers to speed up?
    pub dirty: bool,
    stages: Vec<IllustStage>,
    sam: SuffixTable<'static, 'static>,
    sam_ind: Vec<IllustId>,
}

impl Default for IllustIndex {
    fn default() -> Self {
        Self::parse("[]".to_owned()).unwrap()
    }
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
    s.push_str("\\$s");
    s.push_str(&data.sanity_level.to_string());
    s.push_str("\\$x");
    s.push_str(&data.x_restrict.to_string());
    s.push_str("\\$r");
    s.push_str(&(data.x_restrict + data.sanity_level / 2).to_string());
    if data.sanity_level != 2 {
        s.push_str("\\$n");
    }
    if data.sanity_level >= 6 {
        s.push_str("\\$h");
    }
    if data.width >= data.height {
        s.push_str("\\$w");
    }
    if data.width <= data.height {
        s.push_str("\\$t");
    }
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

        let mut sam = String::new();
        let mut sam_ind = Vec::new();
        let map = HashMap::from_iter(illusts.into_iter().map(|data| {
            // FIXME: do not unwrap
            let o = Illust::new(data).unwrap();
            ids.push(o.data.id);
            sam.push_str(&o.intro);
            for _ in 0..o.intro.len() {
                sam_ind.push(o.data.id);
            }
            (o.data.id, o)
        }));

        let mut stages = Vec::with_capacity(2);
        stages.resize_with(2, Default::default);

        let n = sam.len();
        info!("sam: {n} * 8 = {} MiB", (n * 8) >> 20);

        Ok(Self {
            map,
            ids,
            dirty: false,
            stages,
            sam: SuffixTable::new(sam),
            sam_ind,
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

    pub fn ensure_stage_clean(&self, stage: usize) -> Result<()> {
        let stage = &self.stages[stage];
        if stage.dirty || !stage.ids.is_empty() {
            bail!("stage not clean");
        }
        Ok(())
    }

    pub fn stage(&mut self, stage: usize, illust: Map<String, Value>) -> Result<bool> {
        let illust = Illust::new(illust)?;
        let id = illust.data.id;
        let stage = &mut self.stages[stage];
        if !illust.data.visible {
            if let Some(old) = self.map.get(&id) {
                warn!(
                    "{} ({} - {}) is deleted from upstream, which has luckily been indexed.",
                    id, old.data.title, old.data.user.name
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
            stage.dirty = true;
            return Ok(false);
        }
        stage.dirty = true;
        stage.ids.push(id);
        Ok(true)
    }

    pub fn commit(&mut self, stage: usize) -> usize {
        let stage = &mut self.stages[stage];
        let delta = stage.ids.len();
        let n = self.ids.len();
        self.ids.extend(stage.ids.drain(..).rev());
        assert_eq!(self.ids.len(), n + delta);
        if delta > 0 || stage.dirty {
            self.dirty = true;
        }
        stage.dirty = false;

        self.sam = SuffixTable::new(self.ids.iter().map(|iid| &self.map[iid].intro).join(""));
        self.sam_ind = self
            .ids
            .iter()
            .flat_map(|iid| {
                let n = self.map[iid].intro.len();
                std::iter::repeat(*iid).take(n)
            })
            .collect_vec();

        delta
    }

    pub fn rollback(&mut self, stage: usize) -> usize {
        let stage = &mut self.stages[stage];
        let delta = stage.ids.len();
        let n = self.map.len();
        for id in stage.ids.drain(..) {
            self.map.remove(&id);
        }
        // Callers must ensure ids from different stages don't overlap
        assert_eq!(self.map.len(), n - delta);
        stage.dirty = false;
        delta
    }

    fn single_select(&self, pattern: &str) -> Vec<IllustId> {
        self.sam
            .positions(pattern)
            .iter()
            .sorted_unstable()
            .map(|i| self.sam_ind[*i as usize])
            .dedup()
            .collect_vec()
    }

    fn _select_many_sam(&self, filters: &[String]) -> Vec<IllustId> {
        // assert len(filters) > 0
        let mut sets = filters
            .iter()
            .map(|patt| self.single_select(patt))
            .collect_vec();
        let i = sets
            .iter()
            .enumerate()
            .min_by_key(|(_, v)| v.len())
            .unwrap()
            .0;
        info!("matches: {:?}", sets.iter().map(|v| v.len()).collect_vec());
        let off = sets.split_off(i + 1);
        let mut inter = sets.pop().unwrap();
        for vec in [sets, off] {
            for o in vec {
                let s: HashSet<IllustId, RandomState> = HashSet::from_iter(o.into_iter());
                inter.retain(|e| s.contains(e));
            }
        }
        inter
    }

    fn _select_best_sam(&self, filters: &[String]) -> Vec<&Illust> {
        // assert len(filters) > 0
        let (i, a) = filters
            .iter()
            .map(|patt| self.sam.positions(patt))
            .enumerate()
            .min_by_key(|(_, pos)| pos.len())
            .unwrap();
        a.iter()
            .sorted_unstable()
            .map(|p| self.sam_ind[*p as usize])
            .dedup()
            .map(|iid| &self.map[&iid])
            .filter(|illust| {
                filters
                    .iter()
                    .enumerate()
                    .all(|(p, tag)| p == i || illust.intro.contains(tag))
            })
            .collect_vec()
    }

    pub fn select(&self, filters: &[String]) -> Box<dyn DoubleEndedIterator<Item = &Illust> + '_> {
        if filters.is_empty() {
            return Box::new(self.iter());
        }
        let res = if filters.len() == 1 {
            self.single_select(&filters[0])
        } else {
            // self._select_many_sam(filters)
            return Box::new(self._select_best_sam(filters).into_iter());
        };
        /* assert_eq!(
            res,
            self.iter()
                .filter(|illust| { filters.iter().all(|s| illust.intro.contains(s)) })
                .map(|illust| illust.data.id)
                .collect_vec()
        ); */
        Box::new(res.into_iter().map(|iid| &self.map[&iid]))
    }
}
