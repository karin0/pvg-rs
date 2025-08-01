use crate::bug;
use crate::config::{Config, read_config};
use crate::disk_lru::DiskLru;
use crate::download::{DownloadingFile, DownloadingStream};
use crate::model::{DimCache, Illust, IllustIndex};
use crate::upscale::Upscaler;
use actix_web::web::Bytes;
use anyhow::{Context, Result, bail};
use fs2::FileExt;
use futures::stream::FuturesUnordered;
use futures::{Stream, StreamExt};
use itertools::Itertools;
use parking_lot::{Mutex, RwLock, RwLockWriteGuard};
use pixiv::aapi::Restrict;
use pixiv::client::{AuthedClient, AuthedState};
use pixiv::download::DownloadClient;
use pixiv::{IllustId, PageNum};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashSet;
use std::io;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::{Mutex as TokioMutex, Semaphore, mpsc};
use tokio::time::{Duration, Instant, sleep};
use tokio::{fs, try_join};

#[cfg(feature = "image")]
use crate::model::Dimensions;
#[cfg(feature = "image")]
use image::GenericImageView;
#[cfg(feature = "image")]
use rayon::prelude::*;

#[derive(Deserialize, Default)]
struct LoadedCache {
    token: Option<AuthedState>,
    dims: Option<Vec<(IllustId, Vec<u32>)>>,
    #[serde(default)]
    not_found: HashSet<PathBuf>,
    #[serde(default)]
    worker_to_download: Vec<IllustId>,
}

#[derive(Serialize)]
struct SavingCache<'a> {
    token: &'a AuthedState,
    dims: Vec<(IllustId, Vec<u32>)>,
    not_found: &'a HashSet<PathBuf>,
    worker_to_download: &'a [IllustId],
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
    disk: TokioMutex<()>,
    download_all_lock: TokioMutex<()>,
    not_found: Mutex<HashSet<PathBuf>>,
    db_init_size: usize,
    disk_lru: RwLock<DiskLru>,
    lru_limit: Option<u64>,
    upscaler: Option<Upscaler>,
    update_lock: TokioMutex<()>,
    worker_to_download: TokioMutex<Vec<IllustId>>,
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
        let mut config = read_config()?;
        let refresh_token = std::mem::replace(&mut config.refresh_token, "*".to_owned());
        info!("config: {config:?}");
        if let Some(proxy) = &config.proxy {
            unsafe {
                std::env::set_var("HTTP_PROXY", proxy);
                std::env::set_var("HTTPS_PROXY", proxy);
                std::env::set_var("ALL_PROXY", proxy);
            }
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
                        .write(true)
                        .create_new(true)
                        .open(&config.db_file)?;
                    drop(lock);
                    let lock = std::fs::File::open(&config.db_file)?;
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
        let worker_to_download;
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
                worker_to_download = cache.worker_to_download;
                if let Some(token) = cache.token {
                    info!(
                        "loaded token: {} {} {}",
                        token.user.id, token.user.name, token.user.account
                    );
                    AuthedClient::load(token)
                } else {
                    info!("no cached token");
                    AuthedClient::new(&refresh_token).await?
                }
            }
            Err(e) => {
                if e.kind() == io::ErrorKind::NotFound {
                    info!("no cache file");
                    not_found = Default::default();
                    worker_to_download = Default::default();
                    AuthedClient::new(&refresh_token).await?
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
                warn!("cache size over limit: {total_size} > {limit}");
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

        info!(
            "initialized {} illusts, {} not_founds, {} worker_to_downloads in {} ms",
            db_init_size,
            not_found.len(),
            worker_to_download.len(),
            t.elapsed().as_millis(),
        );
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
            worker_to_download: TokioMutex::new(worker_to_download),
        })
    }

    pub fn is_worker(&self) -> bool {
        self.conf.worker_delay_secs > 0
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

    async fn dump_cache(&self, dims: DimCache) -> serde_json::Result<String> {
        let api = self.api.read().await;
        let worker_to_download = self.worker_to_download.lock().await;
        let token = &api.state;
        let not_found = self.not_found.lock();
        info!(
            "cache: {} {}, {} dims, {} not_founds, {} worker_to_downloads",
            token.user.id,
            token.user.name,
            dims.len(),
            not_found.len(),
            worker_to_download.len()
        );
        let cache = SavingCache {
            token,
            dims,
            not_found: &not_found,
            worker_to_download: &worker_to_download,
        };
        serde_json::to_string(&cache)
    }

    pub async fn dump(&self) -> Result<()> {
        let _ = self.disk.lock().await;
        let dims;
        let mut res = Ok(());
        match self.dump_index() {
            Ok((s, index_dims)) => {
                dims = index_dims;
                if let Some(s) = s {
                    if let Err(e) =
                        fs::rename(&self.conf.db_file, &self.conf.db_file.with_extension("bak"))
                            .await
                    {
                        error!("failed to rename old db: {e}");
                        res = Err(e.into());
                    }
                    info!("writing {} bytes", s.len());
                    match fs::write(&self.conf.db_file, s).await {
                        Ok(_) => info!("written into db {:?}", self.conf.db_file),
                        Err(e) => {
                            error!("failed to write db {:?}: {}", self.conf.db_file, e);
                            res = Err(e.into());
                        }
                    }
                    // TODO: mark index to be clean within the same guard
                } else {
                    info!("index is clean");
                }
            }
            Err(e) => {
                error!("failed to dump index: {e}");
                dims = Default::default();
                res = Err(e);
            }
        }
        let s = self.dump_cache(dims).await?;
        info!("cache: writing {} bytes", s.len());
        tokio::fs::write(&self.conf.cache_file, s).await?;
        info!("cache: written into {:?}", self.conf.cache_file);
        res
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

    async fn _quick_update(&self, stage: usize, restrict: Restrict, pn_limit: u32) -> Result<()> {
        let mut pn = 1;
        if pn > pn_limit {
            return Ok(());
        }
        let mut r: BookmarkPage = self
            .auth()
            .await?
            .user_bookmarks_illust(&self.uid, restrict)
            .await?;
        if !self.is_worker() {
            info!("{:?} page 1: {} illusts", restrict, r.illusts.len());
        }
        if self._quick_update_with_page(stage, r.illusts)? {
            return Ok(());
        }
        while let Some(u) = r.next_url {
            pn += 1;
            if pn > pn_limit {
                break;
            }
            r = loop {
                match self.auth().await?.call_url(&u).await {
                    Ok(x) => {
                        break x;
                    }
                    Err(e) => {
                        if e.to_string().to_ascii_lowercase().contains("rate limit") {
                            warn!("rate limited, sleeping for 60 secs");
                            sleep(Duration::from_secs(60)).await;
                        } else {
                            bail!(e);
                        }
                    }
                }
            };
            if !self.is_worker() {
                info!("page {}: {} illusts", pn, r.illusts.len());
            }
            if self._quick_update_with_page(stage, r.illusts)? {
                break;
            }
        }
        Ok(())
    }

    async fn quick_update_atomic(&self) -> Result<RwLockWriteGuard<IllustIndex>> {
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
            self._quick_update(0, Restrict::Private, limit),
            self._quick_update(1, Restrict::Public, limit),
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
        Ok(index)
    }

    fn commit_stage_worker(
        &self,
        index: &mut RwLockWriteGuard<IllustIndex>,
        stage: usize,
        name: &'static str,
        res: &mut Vec<IllustId>,
    ) {
        let a = index.peek(stage);
        if !a.is_empty() {
            res.extend(a);
            let n = a.len();
            let s = a.iter().map(|i| i.to_string()).join(", ");
            info!("{name}: {n} illusts: {s}");
        }
        index.commit(stage);
    }

    pub async fn quick_update(&self) -> Result<(usize, usize)> {
        let mut index = self.quick_update_atomic().await?;
        // commit private first
        index.commit(0);
        let n_pri = index.commit(0);
        let n_pub = index.commit(1);
        info!("quick updated {n_pub} + {n_pri} illusts");
        Ok((n_pri, n_pub))
    }

    async fn quick_update_worker(&self) -> Result<Vec<IllustId>> {
        let mut ids = Vec::new();
        let mut index = self.quick_update_atomic().await?;
        self.commit_stage_worker(&mut index, 0, "private", &mut ids);
        self.commit_stage_worker(&mut index, 1, "public", &mut ids);
        Ok(ids)
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
        let file = {
            let index = self.index.read();
            let src = &index.get_page(iid, pn).context("no such page")?.source;
            src.filename().to_owned()
        };
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

    async fn disk_remove(&self, file: &str) {
        let path = self.page_path(file);
        if let Err(e) = fs::remove_file(&path).await {
            error!("failed to remove {path:?}: {e}");
        }
    }

    async fn disk_orphan(&self, file: &str) {
        let path = self.page_path(file);
        let dst = self.orphan_path(file);
        if let Err(e) = fs::rename(&path, &dst).await {
            error!("failed to orphan {path:?}: {e}");
        }
    }

    fn disk_evict(&self, limit: u64) -> Option<String> {
        self.disk_lru.write().evict(limit)
    }

    async fn _disk_evict_to(&self, limit: u64) {
        while let Some(file) = self.disk_evict(limit) {
            self.disk_remove(&file).await;
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

    fn orphan_path<T: AsRef<Path>>(&self, file: T) -> PathBuf {
        self.conf.orphan_dir.join(file)
    }

    pub async fn download(
        self: Arc<Self>,
        src: &str,
        path: PathBuf,
    ) -> Result<impl Stream<Item = Result<Bytes>> + use<>> {
        debug!("downloading {src}");
        let mut tmp = self.open_temp(&path).await?;
        let perm = DOWNLOAD_SEMA.acquire().await.unwrap();
        let t = Instant::now();
        let remote = match self.pixiv.download(src).await {
            Err(e) => {
                tmp.rollback().await;
                if let pixiv::Error::Pixiv(404, _) = e {
                    warn!("{path:?}: 404, memorized");
                    self.not_found
                        .lock()
                        .insert(path.file_name().unwrap().into());
                } else {
                    error!("{path:?}: request failed: {e:?}");
                }
                return Err(e.into());
            }
            Ok(r) => r,
        };
        let size = remote.content_length();
        debug!(
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
                        error!("{path:?}: failed to write temp: {e}");
                        drop(perm);
                        tmp.rollback().await;
                        return;
                    }
                } else {
                    drop(perm);
                    match tmp.commit(&path, size).await {
                        Err(e) => error!("{path:?}: failed to save: {e}"),
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
            error!("{path:?}: remote error");
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
                info!("{path:?}: connection failed: {e:?}");
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

    fn make_download_queue(&self) -> Vec<(String, PathBuf)> {
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
            warn!("{cnt_404} pages skipped due to 404");
        }
        r
    }

    fn make_download_queue_from(&self, ids: &[IllustId]) -> Vec<(String, PathBuf)> {
        let index = self.index.read();
        ids.iter()
            .flat_map(|iid| index.map[iid].pages.iter())
            .map(|page| {
                (
                    page.source.url.clone(),
                    self.page_path(page.source.filename()),
                )
            })
            .collect()
    }

    pub async fn download_all(&self) -> Result<i32> {
        let _ = self.download_all_lock.try_lock()?;
        // FIXME: generate the whole queue in sync for now to avoid the use of async locks.
        let q = self.make_download_queue();
        self._download_all(q).await
    }

    async fn download_all_worker(&self, ids: &[IllustId]) -> Result<i32> {
        let _ = self.download_all_lock.try_lock()?;
        let q = self.make_download_queue_from(ids);
        self._download_all(q).await
    }

    async fn _download_all(&self, q: Vec<(String, PathBuf)>) -> Result<i32> {
        if self.lru_limit.is_some() {
            bail!("cache limit is set, refusing to download all");
        }
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
                    error!("{cnt}/{n}: download failed: {e}");
                    // handle 404 (and 500 for 85136899?)
                    if let Ok(pixiv::Error::Pixiv(404, _)) = e.downcast::<pixiv::Error>() {
                        warn!("{path:?}: 404!");
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
            bail!("failed to download {} pages out from {}", cnt_fail, cnt);
        }
        info!("downloaded {cnt} pages");
        Ok(cnt)
    }
}

#[cfg(feature = "image")]
type MeasureQueue = (Vec<(IllustId, Vec<(PageNum, String)>)>, usize);

#[cfg(feature = "image")]
fn measure_file(path: &Path) -> Result<Dimensions> {
    let img = image::open(path)?;
    let (w, h) = img.dimensions();
    Ok(Dimensions(w.try_into()?, h.try_into()?))
}

#[cfg(feature = "image")]
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
    pub fn select(
        &self,
        mut filters: Vec<String>,
        ban_filters: Option<Vec<String>>,
    ) -> serde_json::Result<String> {
        if self.conf.safe_mode {
            filters.push("$s2".to_owned());
        }

        let now = Instant::now();
        let index = self.index.read();
        let it = index.select(&filters);
        let r = if let Some(bans) = ban_filters {
            if !bans.is_empty() {
                self._select(it.filter(|i| !bans.iter().any(|b| i.intro.contains(b))))
            } else {
                self._select(it)
            }
        } else {
            self._select(it)
        };

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

    fn _select<'a>(
        &self,
        index: impl DoubleEndedIterator<Item = &'a Illust>,
    ) -> Vec<SelectedIllust<'a>> {
        // TODO: do this all sync can block for long.
        index
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
            .collect()
    }
}

impl Pvg {
    fn orphan(&self) -> Vec<String> {
        let index = self.index.read();
        let s: HashSet<_> = index
            .iter()
            .flat_map(|illust| illust.pages.iter())
            .map(|page| page.source.filename())
            .collect();

        let mut lru = self.disk_lru.write();
        lru.filter(|file| s.contains(file))
    }

    pub async fn remove_orphans(&self) -> usize {
        let a = self.orphan();
        let n = a.len();
        if n > 0 {
            for file in a {
                warn!("removing orphan: {file}");
                self.disk_remove(&file).await;
            }
            warn!("removed {n} orphans");
        }
        n
    }

    pub async fn move_orphans(&self) -> usize {
        let a = self.orphan();
        let n = a.len();
        if n > 0 {
            for file in a {
                warn!("moving orphan: {file}");
                self.disk_orphan(&file).await;
            }
            info!("moved {n} orphans");
        }
        n
    }

    async fn worker_body(&self) -> Result<()> {
        let ids = self.quick_update_worker().await?;
        let mut todo = self.worker_to_download.lock().await;
        if todo.is_empty() {
            *todo = ids;
        } else {
            warn!("worker: remaining {} illusts to download", todo.len());
            todo.extend(ids);
        }

        if todo.is_empty() {
            // Nothing to do.
            return Ok(());
        }

        // Allow user interruptions while downloading and resume later.
        let ids = todo.clone();
        drop(todo);

        if let Err(e) = self.download_all_worker(&ids).await {
            error!("download_all_worker: {e}");
        }

        // Clear them even if failed to download, as we never retry for now.
        // Some illusts just keep failing with 404/500.
        let mut todo = self.worker_to_download.lock().await;
        todo.clear();

        self.move_orphans().await;
        Ok(())
    }

    // As a worker, we simply poll updates and download them.
    // Disk states should be never checked in worker mode, as the user can move any downloaded stuff away.
    pub async fn worker_start(&self) {
        if self.conf.worker_delay_secs > 0 {
            let d = Duration::from_secs(self.conf.worker_delay_secs as u64);
            info!("worker started with delay {d:?}");
            loop {
                if let Err(e) = self.worker_body().await {
                    error!("worker: {e}");
                }
                sleep(d).await;
            }
        }
    }

    pub async fn qudo(&self) -> Result<()> {
        let (n, m) = self.quick_update().await?;
        if n > 0 || m > 0 {
            let r = self.download_all().await;
            self.move_orphans().await;
            if let Err(e) = r {
                error!("download_all: {e}");
                return Err(e);
            }
        }
        Ok(())
    }
}
