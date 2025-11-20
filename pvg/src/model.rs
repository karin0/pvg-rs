use crate::critical;
use crate::illust::{IllustData, IllustService};
use crate::sa::Query;
use crate::store::Store;
use crate::util::normalized;
use anyhow::{Context, Result, bail};
use itertools::Itertools;
use pixiv::IllustId;
use pixiv::{PageNum, model as api};
use serde::Deserialize;
use serde_json::value::Value as JsonValue;
use std::collections::{BTreeSet, HashMap, hash_map::Entry};
use std::mem::size_of;
use std::num::NonZeroU32;
use std::path::Path;
use std::time::Duration;
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
    pub(crate) data: IllustData,
    pub(crate) pages: Vec<Page>,
    deleted: bool,
}

impl Illust {
    fn memory_url(&self) -> usize {
        self.pages.iter().map(|p| p.source.url.len()).sum::<usize>()
    }

    fn memory(&self) -> usize {
        size_of::<Self>()
            + self
                .pages
                .iter()
                .map(|p| p.source.url.len() + size_of::<Page>())
                .sum::<usize>()
            + self.data.memory()
    }
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
    pub srv: IllustService,
    ids: Vec<IllustId>, // TODO: store pointers to speed up?
    stages: [IllustStage; 2],
    #[cfg(feature = "sam")]
    sa: SAIndex,
    disable_select: bool,
}

impl IllustIndex {
    pub fn stats(&self) {
        let map = self.map.values().map(Illust::memory).sum::<usize>()
            + self.map.len() * size_of::<IllustId>();
        let urls = self.map.values().map(Illust::memory_url).sum::<usize>();
        let ids = self.ids.len() * size_of::<IllustId>();
        let srv = self.srv.memory();
        let srv_lens = self.srv.lens();
        let sa = {
            #[cfg(feature = "sam")]
            {
                self.sa.memory()
            }
            #[cfg(not(feature = "sam"))]
            {
                0
            }
        };
        let tot = map + ids + srv + sa + size_of::<Self>();
        info!(
            "illusts: map={} ({}, {}, {}), srv={}K ({}, {}, {}), sa={}, tot={} MiB",
            map >> 20,
            urls >> 20,
            self.map.len(),
            self.ids.len(),
            srv >> 10,
            srv_lens.0,
            srv_lens.1,
            srv_lens.2,
            sa >> 20,
            tot >> 20
        );
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

fn make_intro(data: &IllustData, srv: &IllustService) -> String {
    let mut s = format!("{}\\{}", data.title, srv.get_user_name(data));
    for t in srv.get_tags(data) {
        s.push('\\');
        s.push_str(t);
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
    fn from_bytes(raw_json: &[u8], srv: &mut IllustService, new: bool) -> Result<Self> {
        let data = serde_json::from_slice::<api::Illust>(raw_json)?;
        Self::from_raw(data, srv, new)
    }

    fn from_raw(data: api::Illust, srv: &mut IllustService, new: bool) -> Result<Self> {
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
            data: srv.resolve(data, new),
            pages,
            deleted: false,
        })
    }

    fn intro(&self, srv: &IllustService) -> String {
        make_intro(&self.data, srv)
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
        let mut srv = IllustService::new();

        let map = illusts
            .into_iter()
            .map(|data| {
                // FIXME: do not unwrap
                let o = Illust::from_bytes(&data, &mut srv, true).unwrap();

                ids.push(o.data.id);

                (o.data.id, o)
            })
            .collect::<HashMap<_, _>>();

        let stats = Store::stats();
        info!(
            "parsed {} illlusts from {} MiB (decompressed from {} MiB in {:.3?})",
            map.len(),
            size >> 20,
            stats.0 >> 20,
            Duration::from_nanos(stats.1 as u64)
        );

        #[cfg(feature = "sam")]
        let sa = if disable_select {
            SAIndex::default()
        } else {
            SAIndex::new(ids.iter().map(|id| (*id, map[id].intro(&srv))))
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
            srv,
            stages: [IllustStage::default(), IllustStage::default()],
            #[cfg(feature = "sam")]
            sa,
            disable_select,
        })
    }

    pub fn load_dims_cache(&mut self, cache: DimCache) -> Result<()> {
        let mut cnt = 0;
        for (iid, a) in cache {
            if let Some(i) = self.map.get_mut(&iid) {
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

    #[cfg(feature = "diff")]
    async fn debug_updated_illust(store: &Store, new: &Illust, item: &StagedItem) -> Result<()> {
        use similar::{ChangeTag, TextDiff};

        let iid = new.data.id;
        let Some(old_json) = store.get_illust(iid).await? else {
            critical!("UPDATED ILLUST {} ABSENT IN DB!\n{:?}", iid, new);
            return Ok(());
        };
        let mut value = serde_json::from_slice::<JsonValue>(&old_json)?;
        drop(old_json);
        value.sort_all_objects();
        let old_json = serde_json::to_string_pretty(&value)?;
        drop(value);

        let mut value = serde_json::from_slice::<JsonValue>(&item.json)?;
        value.sort_all_objects();
        let new_json = serde_json::to_string_pretty(&value)?;
        drop(value);

        let diff = TextDiff::from_lines(&old_json, &new_json);
        let changes = diff
            .iter_all_changes()
            .filter(|c| c.tag() != ChangeTag::Equal)
            .collect_vec();

        if changes.is_empty() {
            // Updated, but not changed. This happens at the first fetch after startup,
            // since `IllustStage.cache` is not persisted.
            // Otherwise, we would be likely to see `total_view` changes every time
            // when we fetch them.
            // As a debugger, we don't prevent the update anyway.
            return Ok(());
        }

        warn!(
            "Illust changed! {iid} ({}): {:?}",
            new.data.title, item.status
        );
        for change in changes {
            match change.tag() {
                ChangeTag::Delete => warn!("- {}", change.to_string_lossy().trim()),
                ChangeTag::Insert => warn!("+ {}", change.to_string_lossy().trim()),
                ChangeTag::Equal => unreachable!(),
            }
        }
        Ok(())
    }

    pub fn stage(&mut self, stage_id: usize, raw: JsonValue) -> Result<bool> {
        // Instead of using `RawValue`, we reserialize it to sanitize the JSON
        // (pixiv escapes unicode chars). Feature `preserve_order` is enforced here.
        let json = serde_json::to_vec(&raw)?;

        let data = serde_json::from_value::<api::Illust>(raw)?;
        let id = data.id;
        let mut illust = Illust::from_raw(data, &mut self.srv, !self.map.contains_key(&id))?;

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
                        debug!("{stage_id}: illust updated: {id}: {}", illust.data.title);
                        #[cfg(not(feature = "diff"))]
                        {
                            debug!("old: {:?}", old.data);
                            debug!("new: {:?}", illust.data);
                        }
                    }
                    #[cfg(feature = "sam")]
                    {
                        let old_intro = old.intro(&self.srv);
                        let intro = illust.intro(&self.srv);
                        status = if old_intro == intro {
                            StagedStatus::IntroUnchanged
                        } else {
                            warn!("{stage_id}: {id}: intro changed (tainting index):");
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
                            id,
                            old.data.title,
                            self.srv.get_original_user_name(&old.data)
                        );
                    } else {
                        warn!(
                            "{} ({} - {}) is deleted from upstream, which sadly hadn't been indexed.",
                            id,
                            old.data.title,
                            self.srv.get_original_user_name(&old.data)
                        );
                    }
                }
                return Ok(false);
            }
            illust.deleted = true;
            warn!(
                "{} ({} - {}) is deleted from upstream, which sadly hasn't been indexed.",
                id,
                illust.data.title,
                self.srv.get_original_user_name(&illust.data)
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

    pub fn count_new(&self, stage_id: usize) -> usize {
        self.stages[stage_id]
            .todo
            .iter()
            .filter(|item| item.status == StagedStatus::New)
            .count()
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

        #[cfg(feature = "diff")]
        if log_enabled!(log::Level::Debug) {
            for item in &stage {
                if item.status != StagedStatus::New {
                    let Some(new) = self.map.get(&item.id) else {
                        critical!("debug_updated_illust: ILLUST {} MISSING IN MAP!", item.id);
                        continue;
                    };
                    if let Err(e) = Self::debug_updated_illust(&self.store, new, item).await {
                        error!("debug_updated_illust: {}: {e:?}", item.id);
                    }
                }
            }
        }

        let new_ids = IllustStage::new_ids(&stage).rev().collect_vec();
        let delta = new_ids.len();
        log!(
            if delta > 0 {
                log::Level::Info
            } else {
                log::Level::Debug
            },
            "{stage_id}: committing {cnt} illusts ({delta} new)"
        );
        if let Err(e) = self
            .store
            .upsert(stage.iter().map(|item| (&item.json, item.id)))
            .await
        {
            critical!("FAILED TO STORE {} ({} new) ILLUSTS: {:?}", cnt, delta, e);
            self.do_rollback(new_ids.into_iter(), stage_id, cnt);

            // Clear the index to alert.
            #[cfg(feature = "sam")]
            if !self.disable_select {
                self.sa = SAIndex::default();
            }

            return 0;
        }

        the_stage.cache = stage.into_iter().map(|item| item.json).collect();
        self.ids.extend(new_ids);

        if !self.disable_select {
            #[cfg(feature = "sam")]
            {
                // XXX: This handles updated illusts with intro changed as well,
                // but the order is not perfectly preserved.
                if !intro_changed_ids.is_empty() {
                    self.sa.insert(
                        intro_changed_ids
                            .iter()
                            .map(|iid| {
                                let o = &self.map[iid];
                                (*iid, o.intro(&self.srv))
                            })
                            .collect_vec(),
                        sa_needs_dedup,
                    );
                }
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

    fn do_select(&self, kind: u32, query: Query) -> Vec<IllustId> {
        let t0 = Instant::now();
        let res = self.sa.select(query);
        let dt = t0.elapsed();
        let st = SAIndex::stats();
        info!(
            "sa{kind}: {} results in {dt:?}, stat {st} ({:.3}/us)",
            res.len(),
            f64::from(st) / (dt.as_secs_f64() * 1e6)
        );
        res
    }

    pub fn select(
        &self,
        filters: &[String],
        ban_filters: &[String],
    ) -> Box<dyn DoubleEndedIterator<Item = &Illust> + '_> {
        if filters.is_empty() && ban_filters.is_empty() {
            return Box::new(self.iter());
        }
        let query = Query::new(filters, ban_filters);

        #[cfg(feature = "sam")]
        {
            #[cfg(feature = "sa-bench")]
            SAIndex::set_flags(0);
            let ans = self.do_select(0, query);

            #[cfg(feature = "sa-bench")]
            {
                // Warm up
                let t0 = Instant::now();
                for kind in 0..5 {
                    SAIndex::set_flags(kind << 1 | 1);
                    self.sa.select(query);
                    SAIndex::stats();
                }
                let dt = t0.elapsed();
                let sum_len = filters
                    .iter()
                    .chain(ban_filters.iter())
                    .map(String::len)
                    .sum::<usize>();
                let num = filters.len() + ban_filters.len();
                info!("warmed up for {num} words, {sum_len} bytes in {dt:?}");
            }

            #[cfg(feature = "sa-bench")]
            for kind in 0..5 {
                SAIndex::set_flags(kind << 1 | 1);

                let res = self.do_select(kind + 1, query);

                if res != ans {
                    critical!("sa{} results differ!", kind);

                    let res = res.into_iter().collect::<BTreeSet<_>>();
                    let ans = ans.iter().copied().collect::<BTreeSet<_>>();

                    let a = res.difference(&ans).collect_vec();
                    let n = a.len();
                    if n > 0 {
                        let s = a.into_iter().take(10).copied().collect_vec();
                        error!("only in res ({n}): {s:?}");
                    }

                    let a = ans.difference(&res).collect_vec();
                    let n = a.len();
                    if n > 0 {
                        let s = a.into_iter().take(10).copied().collect_vec();
                        error!("only in ans ({n}): {s:?}");
                    }
                }
            }

            Box::new(ans.into_iter().map(move |id| &self.map[&id]))
        }
        #[cfg(not(feature = "sam"))]
        {
            error!("not built with sam feature, select is disabled");
            Box::new(std::iter::empty())
        }
    }

    pub fn tags<'a>(&'a self, illust: &'a Illust) -> impl Iterator<Item = &'a str> {
        self.srv.get_tags(&illust.data)
    }

    pub fn user_name(&self, illust: &Illust) -> &str {
        self.srv.get_user_name(&illust.data)
    }
}
