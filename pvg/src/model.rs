use crate::critical;
use crate::store::Store;
use crate::util::normalized;
use anyhow::{Context, Result, bail};
use itertools::Itertools;
use pixiv::IllustId;
use pixiv::{PageNum, model as api};
use serde::Deserialize;
use serde_json::value::Value as JsonValue;
use std::collections::{BTreeSet, HashMap, hash_map::Entry};
use std::num::NonZeroU32;
use std::path::Path;
use tokio::time::Instant;

#[cfg(feature = "sam")]
use crate::sa::Index as SAIndex;

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
    pub(crate) pages: Vec<Page>,
    deleted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StagedStatus {
    New,
    #[cfg(feature = "sam")]
    IntroChanged,
    #[cfg(feature = "sam")]
    IntroUnchanged,
    #[cfg(not(feature = "sam"))]
    Updated,
}

#[derive(Debug, Clone)]
struct StagedItem {
    id: IllustId,
    json: Vec<u8>,
    status: StagedStatus,
}

#[derive(Debug, Default, Clone)]
struct IllustStage {
    cache: BTreeSet<Vec<u8>>,
    todo: Vec<StagedItem>,
}

impl IllustStage {
    fn new_ids(todo: &[StagedItem]) -> impl DoubleEndedIterator<Item = IllustId> + '_ {
        todo.iter().filter_map(|item| {
            if item.status == StagedStatus::New {
                Some(item.id)
            } else {
                None
            }
        })
    }
}

pub struct IllustIndex {
    store: Store,
    pub map: HashMap<IllustId, Illust>,
    ids: Vec<IllustId>, // TODO: store pointers to speed up?
    stages: [IllustStage; 2],
    #[cfg(feature = "sam")]
    sa: SAIndex,
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
    s = normalized(&s);
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
    fn from_bytes(raw_json: &[u8]) -> Result<Self> {
        let data = serde_json::from_slice::<api::Illust>(raw_json)?;
        Self::new(data)
    }

    fn from_value(raw_json: JsonValue) -> Result<Self> {
        let data = serde_json::from_value::<api::Illust>(raw_json)?;
        Self::new(data)
    }

    fn new(data: api::Illust) -> Result<Self> {
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

        Ok(Self {
            data,
            pages,
            deleted: false,
        })
    }

    fn intro(&self) -> String {
        make_intro(&self.data)
    }
}

pub type DimCache = Vec<(IllustId, Vec<u32>)>;

#[derive(Deserialize)]
struct OnlyId {
    id: IllustId,
}

impl IllustIndex {
    pub async fn connect(db_file: &Path, disable_select: bool, create: bool) -> Result<Self> {
        let store = Store::open(db_file, create).await?;
        let illusts = store.illusts().await?;

        if let Ok(s) = std::env::var("PVG_DEBUG_RESAVE_ALL")
            && s == "1"
        {
            warn!("resaving all illusts!");
            let n = illusts.len();
            let todo = illusts.iter().map(|data| {
                let iid = serde_json::from_slice::<OnlyId>(data).unwrap().id;
                (data, iid)
            });
            info!("resaving {n} illusts");
            store.overwrite(todo).await?;
            info!("resaved {n} illusts");
        }

        let size = illusts.iter().map(Vec::len).sum::<usize>();

        let mut ids = Vec::with_capacity(illusts.len());

        let map = illusts
            .into_iter()
            .map(|data| {
                // FIXME: do not unwrap
                let o = Illust::from_bytes(&data).unwrap();

                ids.push(o.data.id);

                (o.data.id, o)
            })
            .collect::<HashMap<_, _>>();

        info!("parsed {} illlusts, raw: {} MiB", map.len(), size >> 20);

        #[cfg(feature = "sam")]
        let sa = if disable_select {
            SAIndex::default()
        } else {
            SAIndex::new(ids.iter().map(|id| (*id, map[id].intro())))
        };

        #[cfg(feature = "sam")]
        {
            let n = sa.size();
            let mem = sa.memory();
            let coef = mem as f64 / n as f64;
            info!("sa: {n} * {coef:.3} = {} MiB", mem >> 20);
        }

        Ok(Self {
            store,
            map,
            ids,
            stages: [IllustStage::default(), IllustStage::default()],
            #[cfg(feature = "sam")]
            sa,
            disable_select,
        })
    }

    pub fn load_dims_cache(&mut self, cache: DimCache) -> Result<()> {
        let mut cnt = 0;
        for (iid, a) in &cache {
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

    pub fn ensure_stage_clean(&self, stage_id: usize) -> Result<()> {
        let stage = &self.stages[stage_id];
        if !stage.todo.is_empty() {
            error!("stage {} is dirty! ({})", stage_id, stage.todo.len());
            bail!("stage not clean");
        }
        Ok(())
    }

    pub fn stage(&mut self, stage_id: usize, raw: JsonValue) -> Result<bool> {
        // Instead of using `RawValue`, we reserialize it to sanitize the JSON
        // (pixiv escapes unicode chars). Feature `preserve_order` is enforced here.
        let json = serde_json::to_vec(&raw)?;

        let mut illust = Illust::from_value(raw)?;
        trace!(
            "staging {} {} {}",
            stage_id, illust.data.id, illust.data.title
        );
        let id = illust.data.id;
        let stage = &mut self.stages[stage_id];
        let mut status = StagedStatus::New;
        if illust.data.visible {
            match self.map.entry(illust.data.id) {
                Entry::Occupied(mut ent) => {
                    let old = ent.get();
                    if !stage.cache.is_empty() {
                        if stage.cache.contains(&json) {
                            // The exactly same data already exists in the map.
                            return Ok(false);
                        }
                        trace!("{stage_id}: illust updated: {id}");
                        trace!("old: {:?}", old.data);
                        trace!("new: {:?}", illust.data);
                    }
                    #[cfg(feature = "sam")]
                    {
                        let old_intro = old.intro();
                        let intro = illust.intro();
                        status = if old_intro == intro {
                            StagedStatus::IntroUnchanged
                        } else {
                            warn!("{stage_id}: intro changed: {id}");
                            warn!("old: {old_intro}");
                            warn!("new: {intro}");
                            StagedStatus::IntroChanged
                        };
                    }
                    #[cfg(not(feature = "sam"))]
                    {
                        status = StagedStatus::Updated;
                    }
                    ent.insert(illust);
                }
                Entry::Vacant(ent) => {
                    ent.insert(illust);
                }
            }
        } else {
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
        }
        stage.todo.push(StagedItem { id, json, status });
        Ok(status == StagedStatus::New)
    }

    pub fn peek(&self, stage_id: usize) -> impl ExactSizeIterator<Item = IllustId> + Clone + '_ {
        self.stages[stage_id].todo.iter().map(|item| item.id)
    }

    // The staged illusts are already applied to `self.map``, we commit them to
    // `self.ids` and the store here.
    pub async fn commit(&mut self, stage_id: usize) -> usize {
        let the_stage = &mut self.stages[stage_id];
        let stage = std::mem::take(&mut the_stage.todo);
        let cnt = stage.len();

        if cnt == 0 {
            return 0;
        }

        #[cfg(feature = "sam")]
        let mut sa_needs_dedup = false;

        #[cfg(feature = "sam")]
        let intro_changed_ids = if self.disable_select {
            Vec::new()
        } else {
            stage
                .iter()
                .filter_map(|item| {
                    if item.status == StagedStatus::IntroUnchanged {
                        None
                    } else {
                        sa_needs_dedup |= item.status == StagedStatus::IntroChanged;
                        Some(item.id)
                    }
                })
                .collect_vec()
        };

        let new_ids = IllustStage::new_ids(&stage).rev().collect_vec();
        let delta = new_ids.len();
        info!("{stage_id}: committing {cnt} illusts ({delta} new)");
        if let Err(e) = self
            .store
            .upsert(stage.iter().map(|item| (&item.json, item.id)))
            .await
        {
            critical!("FAILED TO STORE {} ({} new) ILLUSTS: {:?}", cnt, delta, e);
            self.do_rollback(new_ids.into_iter(), stage_id, cnt);
            return 0;
        }

        the_stage.cache = stage.into_iter().map(|item| item.json).collect();
        self.ids.extend(new_ids);

        if !self.disable_select {
            #[cfg(feature = "sam")]
            {
                // XXX: This handles updated illusts with intro changed as well,
                // but the order is not perfectly preserved.
                self.sa.insert(
                    intro_changed_ids
                        .iter()
                        .map(|iid| {
                            let o = &self.map[iid];
                            (*iid, o.intro())
                        })
                        .collect_vec(),
                    sa_needs_dedup,
                );
            }
        }

        // We return the number of all affected illusts (new + updated) here.
        // This ensures when new pages are added to an existing illust (likely
        // on the first page fetch), the caller still invokes a download task.
        cnt
    }

    pub fn rollback(&mut self, stage_id: usize) -> usize {
        let stage = std::mem::take(&mut self.stages[stage_id].todo);
        let cnt = stage.len();
        self.do_rollback(IllustStage::new_ids(&stage), stage_id, cnt)
    }

    fn do_rollback<I: Iterator<Item = IllustId>>(
        &mut self,
        new_ids: I,
        stage_id: usize,
        cnt: usize,
    ) -> usize {
        let n = self.ids.len();
        for id in new_ids {
            // XXX: The overwritten illust (when new = false) is not restored.
            self.map.remove(&id);
        }
        let delta = n - self.ids.len();
        error!("{stage_id}: rolled back {delta} illusts (out of {cnt})");
        // Callers must ensure ids from different stages don't overlap
        delta
    }

    pub fn select(
        &self,
        filters: &[String],
        ban_filters: &[String],
    ) -> Box<dyn DoubleEndedIterator<Item = &Illust> + '_> {
        if filters.is_empty() && ban_filters.is_empty() {
            return Box::new(self.iter());
        }
        #[cfg(feature = "sam")]
        {
            let t0 = Instant::now();
            let res = self.sa.select(filters, ban_filters);
            info!("sa: {} items in {:?}", res.len(), t0.elapsed());
            Box::new(res.into_iter().map(|iid| &self.map[&iid]))
        }
        #[cfg(not(feature = "sam"))]
        {
            error!("not built with sam feature, select is disabled");
            Box::new(std::iter::empty())
        }
    }
}
