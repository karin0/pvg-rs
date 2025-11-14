use anyhow::{Context, Result, bail};
use itertools::Itertools;
use pixiv::IllustId;
use pixiv::{PageNum, model as api};
use serde::Deserialize;
use serde::de::IntoDeserializer;
use serde_json::{from_str, value::RawValue};
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
    #[cfg(feature = "compress")]
    raw_data: Vec<u8>,
    #[cfg(not(feature = "compress"))]
    raw_data: Box<RawValue>,
    pub(crate) pages: Vec<Page>,
    pub(crate) intro: String,
    deleted: bool,
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
    disable_select: bool,
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
    s = s.to_lowercase();
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
    if data.width >= data.height || data.page_count > 1 {
        s.push_str("\\$l");
    }
    if data.width <= data.height {
        s.push_str("\\$t");
    }
    s
}

impl Illust {
    #[allow(clippy::boxed_local)]
    fn new(raw_data: Box<RawValue>) -> Result<Self> {
        let data = api::Illust::deserialize(raw_data.into_deserializer())?;
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

        #[cfg(feature = "compress")]
        let raw_data = {
            use std::io::Write;

            let raw_data = raw_data.get().as_bytes();
            let mut encoder = lz4::EncoderBuilder::new()
                .content_size(raw_data.len() as u64)
                .build(Vec::new())?;
            encoder.write_all(raw_data)?;
            let (mut raw_data, result) = encoder.finish();
            result?;
            raw_data.shrink_to_fit();
            raw_data
        };

        Ok(Self {
            data,
            raw_data,
            pages,
            intro,
            deleted: false,
        })
    }

    #[cfg(feature = "compress")]
    fn to_raw_data(&self) -> Result<Box<RawValue>> {
        use std::io::Read;

        let mut decoder = lz4::Decoder::new(&self.raw_data[..])?;
        let mut s = String::new();
        decoder.read_to_string(&mut s)?;
        Ok(RawValue::from_string(s)?)
    }
}

pub type DimCache = Vec<(IllustId, Vec<u32>)>;

impl IllustIndex {
    pub fn parse(s: String, disable_select: bool) -> serde_json::error::Result<Self> {
        let illusts: Vec<Box<RawValue>> = from_str(&s)?;
        let size = illusts.iter().map(|i| i.get().len()).sum::<usize>();
        let mut ids = Vec::with_capacity(illusts.len());

        let mut sam = String::new();
        let mut sam_ind = Vec::new();
        let map = HashMap::from_iter(illusts.into_iter().map(|data| {
            // FIXME: do not unwrap
            let o = Illust::new(data).unwrap();
            ids.push(o.data.id);
            if !disable_select {
                sam.push_str(&o.intro);
                for _ in 0..o.intro.len() {
                    sam_ind.push(o.data.id);
                }
            }
            (o.data.id, o)
        }));
        debug!("parsed {} illlusts", map.len());

        let n = sam.len();
        #[cfg(feature = "compress")]
        {
            let size2 = map.values().map(|i| i.raw_data.len()).sum::<usize>();
            info!(
                "raw: {} MiB -> {} MiB, sam: {n} * 8 = {} MiB",
                size >> 20,
                size2 >> 20,
                n >> 17
            );
        }

        #[cfg(not(feature = "compress"))]
        info!("raw: {} MiB, sam: {n} * 8 = {} MiB", size >> 20, n >> 17);

        let mut stages = Vec::with_capacity(2);
        stages.resize_with(2, Default::default);

        let sam = SuffixTable::new(sam);
        debug!("built suffix table");

        Ok(Self {
            map,
            ids,
            dirty: false,
            stages,
            sam,
            sam_ind,
            disable_select,
        })
    }

    pub fn dump(&self) -> Result<Vec<u8>> {
        // FIXME: serde can't do async streaming for now.
        info!("collecting {} illlusts", self.len());
        #[cfg(feature = "compress")]
        let v = self
            .iter()
            .map(|i| i.to_raw_data())
            .collect::<Result<Vec<_>>>()?;

        #[cfg(not(feature = "compress"))]
        let v = self.iter().map(|i| &i.raw_data).collect::<Vec<_>>();
        info!("dumping {} illlusts", v.len());
        Ok(serde_json::to_vec(&v)?)
    }

    pub fn load_dims_cache(&mut self, cache: DimCache) -> Result<()> {
        let mut cnt = 0;
        for (iid, a) in cache.iter() {
            if let Some(i) = self.map.get_mut(iid) {
                cnt += 1;
                let pc = i.data.page_count as usize;
                if (pc - 1) * 2 == a.len() {
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
                warn!("no such illust: {iid}");
            }
        }
        info!("loaded {cnt} dimensions from cache");
        Ok(())
    }

    #[allow(dead_code)]
    fn dump_dims_cache(&self) -> Vec<(IllustId, Vec<u32>)> {
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

    pub fn get_page(&self, iid: IllustId, pn: PageNum) -> Option<&Page> {
        self.map.get(&iid).and_then(|i| i.pages.get(pn as usize))
    }

    pub fn ensure_stage_clean(&self, stage: usize) -> Result<()> {
        let stage = &self.stages[stage];
        if stage.dirty || !stage.ids.is_empty() {
            bail!("stage not clean");
        }
        Ok(())
    }

    pub fn stage(&mut self, stage: usize, illust: Box<RawValue>) -> Result<bool> {
        let mut illust = Illust::new(illust)?;
        let id = illust.data.id;
        let stage = &mut self.stages[stage];
        if !illust.data.visible {
            if let Some(old) = self.map.get_mut(&id) {
                if !old.deleted {
                    old.deleted = true;
                    if old.data.visible {
                        warn!(
                            "{} ({} - {}) is deleted from upstream, which has luckily been indexed.",
                            id, old.data.title, old.data.user.name
                        );
                    } else {
                        warn!(
                            "{} ({} - {}) is deleted from upstream, which sadly hadn't been indexed.",
                            id, old.data.title, old.data.user.name
                        );
                    }
                }
                return Ok(false);
            }
            illust.deleted = true;
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

    pub fn peek(&self, stage: usize) -> &[IllustId] {
        &self.stages[stage].ids
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

        if !self.disable_select {
            self.sam = SuffixTable::new(self.ids.iter().map(|iid| &self.map[iid].intro).join(""));
            self.sam_ind = self
                .ids
                .iter()
                .flat_map(|iid| {
                    let n = self.map[iid].intro.len();
                    std::iter::repeat_n(*iid, n)
                })
                .collect_vec();
        }

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
