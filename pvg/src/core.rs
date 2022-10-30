use crate::bug;
use crate::config::{read_config, Config};
use crate::disk_lru::DiskLru;
use crate::download::{DownloadingFile, DownloadingStream};
use crate::model::{DimCache, Dimensions, IllustIndex};
use crate::upscale::Upscaler;
use actix_web::web::Bytes;
use anyhow::{bail, Context, Result};
use fs2::FileExt;
use futures::stream::FuturesUnordered;
use futures::{Stream, StreamExt};
use image::GenericImageView;
use parking_lot::{Mutex, RwLock};
use pixiv::aapi::BookmarkRestrict;
use pixiv::client::{AuthedClient, AuthedState};
use pixiv::download::DownloadClient;
use pixiv::{IllustId, PageNum};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashSet;
use std::io;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::{mpsc, Semaphore};
use tokio::time::Instant;
use tokio::{fs, try_join};

#[derive(Deserialize, Default)]
struct LoadedCache {
    token: Option<AuthedState>,
    dims: Option<Vec<(IllustId, Vec<u32>)>>,
    #[serde(default)]
    not_found: HashSet<PathBuf>,
}

#[derive(Serialize)]
struct SavingCache<'a> {
    token: &'a AuthedState,
    dims: Vec<(IllustId, Vec<u32>)>,
    not_found: &'a HashSet<PathBuf>,
}

#[derive(Debug)]
pub struct Pvg {
    pub conf: Config,
    #[allow(dead_code)]
    lock: std::fs::File,
    index: RwLock<IllustIndex>,
    api: tokio::sync::RwLock<AuthedClient>,
    pixiv: DownloadClient,
    uid: String,
    disk: tokio::sync::Mutex<()>,
    download_all_lock: tokio::sync::Mutex<()>,
    not_found: Mutex<HashSet<PathBuf>>,
    db_init_size: usize,
    disk_lru: RwLock<DiskLru>,
    lru_limit: Option<u64>,
    upscaler: Option<Upscaler>,
    update_lock: tokio::sync::Mutex<()>,
}

#[derive(Deserialize, Debug)]
pub struct BookmarkPage {
    illusts: Vec<Map<String, Value>>,
    next_url: Option<String>,
}

static DOWNLOAD_SEMA: Semaphore = Semaphore::const_new(20);

impl Pvg {
    pub async fn new() -> Result<Pvg> {
        let t = Instant::now();
        let config = read_config()?;
        info!("config: {:?}", config);
        if let Some(proxy) = &config.proxy {
            std::env::set_var("HTTP_PROXY", proxy);
            std::env::set_var("HTTPS_PROXY", proxy);
            std::env::set_var("ALL_PROXY", proxy);
        }
        let (mut nav, lock) = match std::fs::File::open(&config.db_file) {
            Ok(mut db) => {
                db.try_lock_exclusive()?;
                let mut s = String::new();
                db.read_to_string(&mut s)?;
                info!("read {} bytes from {:?}", s.len(), db);
                let nav = IllustIndex::parse(s, config.disable_select)?;
                (nav, db)
            }
            Err(e) => {
                if e.kind() == io::ErrorKind::NotFound {
                    info!("creating new db file");
                    let lock = std::fs::OpenOptions::new()
                        .create_new(true)
                        .open(&config.db_file)?;
                    lock.try_lock_exclusive()?;
                    (
                        IllustIndex::parse("[]".to_owned(), config.disable_select)?,
                        lock,
                    )
                } else {
                    return Err(e.into());
                }
            }
        };
        info!(
            "index: {} illusts in {} ms",
            nav.len(),
            t.elapsed().as_millis()
        );

        let not_found;
        let api = match std::fs::File::open(&config.cache_file) {
            Ok(mut cache) => {
                let mut s = String::new();
                cache.read_to_string(&mut s)?;
                info!("read {} bytes from {:?}", s.len(), cache);
                let cache: LoadedCache = serde_json::from_str(&s)?;
                if let Some(dims) = cache.dims {
                    nav.load_dims_cache(dims)?;
                } else {
                    info!("no cached dims");
                }
                not_found = cache.not_found;
                if let Some(token) = cache.token {
                    info!(
                        "loaded token: {} {} {}",
                        token.user.id, token.user.name, token.user.account
                    );
                    AuthedClient::load(token)
                } else {
                    info!("no cached token");
                    AuthedClient::new(&config.refresh_token).await?
                }
            }
            Err(e) => {
                if e.kind() == io::ErrorKind::NotFound {
                    info!("no cache file");
                    not_found = HashSet::new();
                    AuthedClient::new(&config.refresh_token).await?
                } else {
                    return Err(e.into());
                }
            }
        };

        info!("api: {} {}", api.state.user.name, api.state.user.id);
        let uid = api.state.user.id.to_string();

        let mut vec = Vec::with_capacity(nav.len());
        let mut total_size = 0;
        let mut files = fs::read_dir(&config.pix_dir).await?;
        while let Some(file) = files.next_entry().await? {
            let meta = file.metadata().await?;
            let file = file.file_name().into_string().unwrap();
            let size = meta.len();
            let time = if config.cache_limit.is_some() {
                meta.accessed()
                    .or_else(|_| meta.modified())
                    .or_else(|_| meta.created())
                    .unwrap_or(SystemTime::UNIX_EPOCH)
            } else {
                SystemTime::UNIX_EPOCH
            };
            vec.push((file, size, time));
            total_size += size;
        }
        info!(
            "disk: {} files, {:.2} MiB",
            vec.len(),
            total_size as f32 / ((1 << 20) as f32)
        );
        let lru_limit = if let Some(limit) = config.cache_limit {
            vec.sort_unstable_by_key(|(_, _, time)| time.to_owned());
            if total_size > limit {
                warn!("cache size over limit: {} > {}", total_size, limit);
                Some(total_size)
            } else {
                Some(limit)
            }
        } else {
            None
        };

        let mut lru = DiskLru::new();
        for (file, size, _) in vec {
            lru.insert(file, size);
        }

        let db_init_size = nav.len();
        let upscaler = if let Some(ref path) = config.upscaler_path {
            Some(Upscaler::new(path.clone(), &config).await?)
        } else {
            None
        };
        Ok(Pvg {
            conf: config,
            lock,
            index: RwLock::new(nav),
            api: tokio::sync::RwLock::new(api),
            uid,
            pixiv: DownloadClient::new(),
            download_all_lock: Default::default(),
            disk: Default::default(),
            not_found: Mutex::new(not_found),
            db_init_size,
            disk_lru: RwLock::new(lru),
            lru_limit,
            upscaler,
            update_lock: Default::default(),
        })
    }

    fn dump_index(&self) -> Result<(Option<Vec<u8>>, DimCache)> {
        let index = self.index.read();
        let db;
        if index.dirty {
            db = Some(index.dump()?);
        } else {
            let n = index.len();
            if n != self.db_init_size {
                bug!(
                    "DB SIZE CHANGED WITHOUT DIRTY FLAG! {} -> {}",
                    self.db_init_size,
                    n
                );
                db = Some(index.dump()?);
            } else {
                db = None;
            }
        }
        let dims = index.dump_dims_cache();
        Ok((db, dims))
    }

    fn dump_cache(&self, dims: DimCache, token: &AuthedState) -> serde_json::Result<String> {
        let not_found = self.not_found.lock();
        info!(
            "cache: {} {}, {} dims, {} not_founds",
            token.user.id,
            token.user.name,
            dims.len(),
            not_found.len()
        );
        let cache = SavingCache {
            token,
            dims,
            not_found: &not_found,
        };
        serde_json::to_string(&cache)
    }

    pub async fn dump(&self) -> Result<()> {
        let (s, dims) = self.dump_index()?;
        let _ = self.disk.lock().await;
        if let Some(s) = s {
            fs::rename(&self.conf.db_file, &self.conf.db_file.with_extension("bak")).await?;
            info!("writing {} bytes", s.len());
            fs::write(&self.conf.db_file, s).await?;
            info!("written into {:?}", self.conf.db_file);
            // TODO: mark index to be clean within the same guard
        } else {
            info!("index is clean");
        }
        let api = self.api.read().await;
        let s = self.dump_cache(dims, &api.state)?;
        drop(api);
        info!("cache: writing {} bytes", s.len());
        tokio::fs::write(&self.conf.cache_file, s).await?;
        info!("cache: written into {:?}", self.conf.cache_file);
        Ok(())
    }

    fn _quick_update_with_page(&self, stage: usize, r: Vec<Map<String, Value>>) -> Result<bool> {
        let mut updated = false;
        let mut index = self.index.write();
        for illust in r {
            if index.stage(stage, illust)? {
                updated = true;
            }
        }
        Ok(!updated)
    }

    async fn auth(&self) -> Result<tokio::sync::RwLockReadGuard<AuthedClient>> {
        {
            let mut api = self.api.write().await;
            api.ensure_authed().await?;
        }
        Ok(self.api.read().await)
    }

    async fn _quick_update(
        &self,
        stage: usize,
        restrict: BookmarkRestrict,
        pn_limit: u32,
    ) -> Result<()> {
        let mut pn = 1;
        if pn > pn_limit {
            return Ok(());
        }
        let mut r: BookmarkPage = self
            .auth()
            .await?
            .user_bookmarks_illust(&self.uid, restrict)
            .await?;
        info!("{:?} page 1: {} illusts", restrict, r.illusts.len());
        if self._quick_update_with_page(stage, r.illusts)? {
            return Ok(());
        }
        while let Some(u) = r.next_url {
            pn += 1;
            if pn > pn_limit {
                break;
            }
            r = self.auth().await?.call_url(&u).await?;
            info!("page {}: {} illusts", pn, r.illusts.len());
            if self._quick_update_with_page(stage, r.illusts)? {
                break;
            }
        }
        Ok(())
    }

    pub async fn quick_update(&self) -> Result<()> {
        let _ = self.update_lock.try_lock()?;
        let empty = {
            let index = self.index.read();
            index.ensure_stage_clean(0)?;
            index.ensure_stage_clean(1)?;
            index.len() == 0
        };
        let limit = if empty {
            self.conf.first_time_pn_limit.unwrap_or(u32::MAX)
        } else {
            u32::MAX
        };
        let r = try_join!(
            self._quick_update(0, BookmarkRestrict::Private, limit),
            self._quick_update(1, BookmarkRestrict::Public, limit),
        );
        let mut index = self.index.write();
        if let Err(e) = r {
            error!(
                "quick update failed ({} + {} rolled back): {}",
                index.rollback(1),
                index.rollback(0),
                e,
            );
            return Err(e);
        }
        // commit private first
        let n = index.commit(0);
        info!("quick updated {} + {} illusts", index.commit(1), n);
        Ok(())
    }

    pub fn get_source(&self, iid: IllustId, pn: PageNum) -> Option<(String, PathBuf)> {
        let index = self.index.read();
        let src = &index.get_page(iid, pn)?.source;
        let file = src.filename();
        let pile: &Path = file.as_ref();
        if self.not_found.lock().contains(pile) {
            return None;
        }
        if self.lru_limit.is_some() {
            self.disk_lru.write().promote(file);
        }
        let url = src.url.clone();
        let path = self.page_path(pile);
        Some((url, path))
    }

    pub async fn upscale(&self, iid: IllustId, pn: PageNum, scale: u8) -> Result<PathBuf> {
        let index = self.index.read();
        let src = &index.get_page(iid, pn).context("no such page")?.source;
        let file = src.filename();
        let out_file = format!("{}_{}x.png", file.replace('.', "_"), scale);
        let out_path = self.conf.upscale_dir.join(&out_file);
        if fs::metadata(&out_path).await.is_ok() {
            return Ok(out_path);
        }
        // TODO: manage space of upscaled files in lru cache
        // TODO: download if source doesn't exist
        // TODO: handle if already started
        let upscaler = self.upscaler.as_ref().context("no upscaler configured")?;
        upscaler.run(file, out_file, scale).await?;
        Ok(out_path)
    }

    async fn open_temp(&self, path: &Path) -> Result<DownloadingFile> {
        let tmp_path = self.conf.tmp_dir.join(path.file_name().unwrap());
        // TODO: this fails if a file is requested twice before it's downloaded, do some waiting.
        DownloadingFile::new(tmp_path).await
    }

    fn disk_evict(&self, limit: u64) -> Option<String> {
        self.disk_lru.write().evict(limit)
    }

    async fn _disk_evict_to(&self, limit: u64) {
        while let Some(file) = self.disk_evict(limit) {
            let path = self.page_path(file);
            if let Err(e) = fs::remove_file(&path).await {
                error!("failed to remove {:?}: {}", path, e);
            }
        }
    }

    async fn disk_evict_all(&self) {
        if let Some(limit) = self.lru_limit {
            self._disk_evict_to(limit).await;
        }
    }

    pub async fn enforce_cache_limit(&self) {
        if let Some(limit) = self.conf.cache_limit {
            self._disk_evict_to(limit).await;
        }
    }

    fn page_path<T: AsRef<Path>>(&self, file: T) -> PathBuf {
        self.conf.pix_dir.join(file)
    }

    pub async fn download(
        self: Arc<Self>,
        src: &str,
        path: PathBuf,
    ) -> Result<impl Stream<Item = Result<Bytes>>> {
        info!("downloading {}", src);
        let mut tmp = self.open_temp(&path).await?;
        let perm = DOWNLOAD_SEMA.acquire().await.unwrap();
        let t = Instant::now();
        let remote = match self.pixiv.download(src).await {
            Err(e) => {
                tmp.rollback().await;
                if let pixiv::Error::Pixiv(404, _) = e {
                    warn!("{:?}: 404, memorized", path);
                    self.not_found
                        .lock()
                        .insert(path.file_name().unwrap().into());
                } else {
                    error!("{:?}: request failed: {:?}", path, e);
                }
                return Err(e.into());
            }
            Ok(r) => r,
        };
        let size = remote.content_length();
        info!(
            "{:?}: connection established in {:.3} secs, {:?} B",
            path,
            t.elapsed().as_secs_f32(),
            size
        );
        tmp.start();

        let (tx, mut rx) = mpsc::unbounded_channel::<Option<Bytes>>();

        let npath = path.clone();
        tokio::spawn(async move {
            let path = npath;
            let perm = perm;
            while let Some(msg) = rx.recv().await {
                if let Some(b) = msg {
                    if let Err(e) = tmp.write(&b).await {
                        error!("{:?}: failed to write temp: {}", path, e);
                        drop(perm);
                        tmp.rollback().await;
                        return;
                    }
                } else {
                    drop(perm);
                    match tmp.commit(&path, size).await {
                        Err(e) => error!("{:?}: failed to save: {}", path, e),
                        Ok(size) => {
                            let file = path.file_name().unwrap().to_str().unwrap();
                            {
                                self.disk_lru.write().insert(file.to_string(), size);
                            }
                            self.disk_evict_all().await;
                        }
                    }
                    return;
                }
            }
            error!("{:?}: remote error", path);
            drop(perm);
            tmp.rollback().await;
        });

        Ok((DownloadingStream { remote, path, tx }).stream())
    }

    async fn _download_file(
        mut resp: pixiv::reqwest::Response,
        file: &mut DownloadingFile,
    ) -> Result<()> {
        while let Some(b) = resp.chunk().await? {
            file.write(&b).await?;
        }
        Ok(())
    }

    async fn _downloader_inner(&self, url: &str, path: &Path) -> Result<u64> {
        let perm = DOWNLOAD_SEMA.acquire().await?;
        let mut tmp = self.open_temp(path).await?;
        let r = match self.pixiv.download(url).await {
            Err(e) => {
                tmp.rollback().await;
                info!("{:?}: connection failed: {:?}", path, e);
                return Err(e.into());
            }
            Ok(r) => r,
        };
        let size = r.content_length();

        tmp.start();
        let res = Self::_download_file(r, &mut tmp).await;
        drop(perm);
        if let Err(e) = res {
            tmp.rollback().await;
            return Err(e);
        }
        tmp.commit(path, size).await
    }

    async fn _downloader(&self, url: String, path: PathBuf) -> (PathBuf, Result<u64>) {
        let r = self._downloader_inner(&url, &path).await;
        (path, r)
    }

    fn _make_download_queue(&self) -> Vec<(String, PathBuf)> {
        let not_found = self.not_found.lock();
        let disk = self.disk_lru.read();
        let mut cnt_404 = 0;
        let r = self
            .index
            .read()
            .iter()
            .flat_map(|illust| illust.pages.iter())
            .map(|page| (&page.source, page.source.filename()))
            .filter(|(_, file)| {
                if !disk.contains(file) {
                    let file: &Path = file.as_ref();
                    if !not_found.contains(file) {
                        return true;
                    } else {
                        cnt_404 += 1;
                    }
                }
                false
            })
            .map(|(src, file)| (src.url.clone(), self.page_path(file)))
            .collect();
        if cnt_404 > 0 {
            warn!("{} pages skipped due to 404", cnt_404);
        }
        r
    }

    pub async fn download_all(&self) -> Result<()> {
        if self.lru_limit.is_some() {
            bail!("cache limit is set, refusing to download all");
        }
        // FIXME: generate the whole queue in sync for now to avoid the use of async locks.
        let _ = self.download_all_lock.try_lock()?;
        let t = Instant::now();
        let q = self._make_download_queue();
        warn!("download_all: blocked for {} ms", t.elapsed().as_millis());
        info!("{} pages to download", q.len());
        let mut futs = q
            .into_iter()
            .map(|(url, path)| self._downloader(url, path))
            .collect::<FuturesUnordered<_>>();
        let n = futs.len();
        let mut cnt = 0;
        let mut cnt_fail = 0;
        let mut the_404 = vec![];
        while let Some((path, res)) = futs.next().await {
            cnt += 1;
            match res {
                Ok(size) => {
                    info!("{}/{}: downloaded {:?} ({} KiB)", cnt, n, path, size >> 10);
                    self.disk_lru
                        .write()
                        .insert(path.file_name().unwrap().to_str().unwrap().to_owned(), size);
                }
                Err(e) => {
                    cnt_fail += 1;
                    error!("{}/{}: download failed: {}", cnt, n, e);
                    // handle 404 (and 500 for 85136899?)
                    if let Ok(pixiv::Error::Pixiv(404, _)) = e.downcast::<pixiv::Error>() {
                        warn!("{:?}: 404!", path);
                        the_404.push(path.file_name().unwrap().into());
                    }
                }
            }
        }
        drop(futs);
        if !the_404.is_empty() {
            warn!("collecting {} not_founds", the_404.len());
            self.not_found.lock().extend(the_404.into_iter());
        }
        if cnt_fail > 0 {
            bail!("{} downloads failed", cnt_fail);
        }
        Ok(())
    }
}

type MeasureQueue = (Vec<(IllustId, Vec<(PageNum, String)>)>, usize);

fn measure_file(path: &Path) -> Result<Dimensions> {
    let img = image::open(path)?;
    let (w, h) = img.dimensions();
    Ok(Dimensions(w.try_into()?, h.try_into()?))
}

impl Pvg {
    fn _make_measure_queue(&self) -> MeasureQueue {
        let mut vec = vec![];
        let mut n = 0;
        let r: Vec<(_, Vec<(PageNum, String)>)> = self
            .index
            .read()
            .iter()
            .filter_map(|illust| {
                vec.clear();
                for (i, page) in illust.pages.iter().skip(1).enumerate() {
                    if page.dimensions.is_none() {
                        vec.push((i as PageNum + 1, page.source.filename().to_owned()));
                    }
                }
                if vec.is_empty() {
                    None
                } else {
                    n += vec.len();
                    Some((illust.data.id, vec.clone()))
                }
            })
            .collect();
        info!("{} illusts has {} pages to measure", r.len(), n);
        (r, n)
    }

    fn _set_dims(&self, iid: IllustId, pn: PageNum, dims: Dimensions) -> Result<()> {
        self.index
            .write()
            .map
            .get_mut(&iid)
            .context("no such illust")?
            .pages
            .get_mut(pn as usize)
            .context("no such page")?
            .dimensions = Some(dims);
        Ok(())
    }

    pub async fn measure_all(&self) -> Result<()> {
        // let _ = self.measure_all_lock.try_lock()?;
        let t = Instant::now();
        let (q, n) = self._make_measure_queue();
        warn!("measure_all: blocked for {} ms", t.elapsed().as_millis());
        info!("{} illusts to measure", q.len());

        let (tx, mut rx) = mpsc::unbounded_channel();
        let base = self.conf.pix_dir.clone();
        rayon::spawn(move || {
            q.into_par_iter()
                .flat_map(|(iid, vec)| vec.into_par_iter().map(move |(pn, v)| (iid, pn, v)))
                .for_each(move |(iid, pn, filename)| {
                    let file = base.join(filename);
                    let r = measure_file(&file);
                    if let Err(e) = tx.send((iid, pn, r)) {
                        bug!("{:?}: send error: {}", file, e);
                    }
                });
        });

        let mut cnt = 0;
        let mut cnt_fail = 0;
        while let Some((iid, pn, res)) = rx.recv().await {
            cnt += 1;
            match res {
                Ok(dims) => match self._set_dims(iid, pn, dims) {
                    Ok(_) => info!("{}/{}: measured {:?}", cnt, n, dims),
                    Err(e) => {
                        cnt_fail += 1;
                        error!("{}/{}: set dims failed: {}", cnt, n, e);
                    }
                },
                Err(e) => {
                    cnt_fail += 1;
                    error!("{}/{}: measure failed: {}", cnt, n, e);
                }
            }
        }
        if cnt_fail > 0 {
            bail!("{} measures failed", cnt_fail);
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
struct SelectedPage<'a>(u32, u32, &'a str, &'a str);

#[derive(Debug, Serialize)]
struct SelectedIllust<'a>(
    IllustId,
    &'a str,
    u32,
    &'a str,
    Vec<&'a str>,
    Vec<SelectedPage<'a>>,
    &'a str,
    u16,
);

#[derive(Debug, Serialize)]
struct SelectResponse<'a> {
    items: Vec<SelectedIllust<'a>>,
}

impl Pvg {
    pub fn select(&self, filters: &[String]) -> serde_json::Result<String> {
        // TODO: do this all sync can block for long.
        let now = Instant::now();
        let index = self.index.read();
        let r: Vec<SelectedIllust> = index
            .select(filters)
            .rev()
            .map(|illust| {
                let tags: Vec<&str> = illust.data.tags.iter().map(|t| t.name.as_ref()).collect();
                let mut curr_w = 0;
                let mut curr_h = 0;
                let pages: Vec<SelectedPage> = illust
                    .pages
                    .iter()
                    .map(|page| {
                        if let Some(d) = page.dimensions {
                            curr_w = d.0.get();
                            curr_h = d.1.get();
                        };
                        SelectedPage(curr_w, curr_h, "img", page.source.filename())
                    })
                    .collect();
                let i = &illust.data;
                /* if unmeasured > 0 {
                    warn!(
                        "{}: {} unmeasured pages (of {})",
                        i.id, unmeasured, i.page_count
                    );
                } */
                SelectedIllust(
                    i.id,
                    &i.title,
                    i.user.id,
                    &i.user.name,
                    tags,
                    pages,
                    &i.create_date,
                    (i.sanity_level >> 1) + i.x_restrict,
                )
            })
            .collect();
        let now2 = Instant::now();
        let t = (now2 - now).as_millis();
        let n = r.len();
        let r = serde_json::to_string(&SelectResponse { items: r })?;
        info!(
            "{:?}: {} illusts ({:.1} KiB) in {} + {} ms",
            filters,
            n,
            r.len() as f32 / 1024.,
            t,
            now2.elapsed().as_millis()
        );
        Ok(r)
    }
}
