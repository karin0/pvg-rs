use crate::config::{Config, read_config};
use crate::disk_lru::DiskLru;
use crate::download::{DownloadingFile, DownloadingStream};
use crate::model::{Dimension, Illust, IllustIndex};
use crate::upscale::Upscaler;
use actix_web::web::Bytes;
use anyhow::{Context, Result, bail};
use fs2::FileExt;
use futures::stream::FuturesUnordered;
use futures::{Stream, StreamExt};
use itertools::Itertools;
use parking_lot::{Mutex, RwLock};
use pixiv::aapi::Restrict;
use pixiv::client::{AuthedClient, AuthedState};
use pixiv::download::DownloadClient;
use pixiv::{IllustId, PageNum};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::value::Value;
use std::collections::HashSet;
use std::fs::File;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{Mutex as TokioMutex, RwLock as TokioRwLock, RwLockWriteGuard, Semaphore, mpsc};
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
    dims: Option<Vec<(IllustId, Vec<Dimension>)>>,
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

pub struct Pvg {
    pub conf: Config,
    index: TokioRwLock<IllustIndex>,
    api: TokioRwLock<AuthedClient>,
    pixiv: DownloadClient,
    uid: String,
    download_all_lock: TokioMutex<()>,
    not_found: Mutex<HashSet<PathBuf>>,
    disk_lru: RwLock<DiskLru>,
    lru_limit: Option<u64>,
    upscaler: Option<Upscaler>,
    update_lock: TokioMutex<()>,
    worker_to_download: TokioMutex<Vec<IllustId>>,
}

#[derive(Deserialize, Debug)]
pub struct BookmarkPage {
    illusts: Vec<Value>,
    next_url: Option<String>,
}

static DOWNLOAD_SEMA: Semaphore = Semaphore::const_new(20);

fn lock_file<T: DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    let lock;
    let r = match File::open(path) {
        Ok(file) => {
            file.try_lock_exclusive()?;
            let reader = std::io::BufReader::new(file);
            let r = Some(serde_json::from_reader::<_, T>(reader)?);
            lock = File::open(path)?;
            r
        }
        Err(e) => {
            if e.kind() == ErrorKind::NotFound {
                info!("{}: not exist, creating", path.display());
                let file = File::create(path)?;
                lock = file;
                None
            } else {
                return Err(e.into());
            }
        }
    };
    lock.try_lock_exclusive()?;
    std::mem::forget(lock); // keep the lock held until process exit
    Ok(r)
}

impl Pvg {
    pub async fn new() -> Result<Pvg> {
        let t = Instant::now();
        let mut config = read_config()?;
        let refresh_token = std::mem::replace(&mut config.refresh_token, "*".to_owned());
        info!("config: {config:?}");

        let cache = lock_file::<LoadedCache>(&config.cache_file)?;

        if let Some(proxy) = &config.proxy {
            unsafe {
                std::env::set_var("HTTP_PROXY", proxy);
                std::env::set_var("HTTPS_PROXY", proxy);
                std::env::set_var("ALL_PROXY", proxy);
            }
        }
        let mut nav = IllustIndex::connect(
            config.db_file.as_path(),
            config.disable_select,
            config.worker_delay_secs == 0,
        )
        .await?;
        info!(
            "index: {} illusts in {} ms",
            nav.len(),
            t.elapsed().as_millis()
        );
        nav.stats();

        let not_found;
        let worker_to_download;
        let api = if let Some(cache) = cache {
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
        } else {
            not_found = HashSet::new();
            worker_to_download = Vec::new();
            AuthedClient::new(&refresh_token).await?
        };

        info!("api: {} {}", api.state.user.name, api.state.user.id);
        let uid = api.state.user.id.to_string();

        let index_size = nav.len();
        let pages = nav.count_pages();
        let (lru, lru_limit) = DiskLru::load(pages, &config.pix_dir, config.cache_limit).await?;

        let upscaler = if let Some(ref path) = config.upscaler_path {
            Some(Upscaler::new(path.clone(), &config).await?)
        } else {
            None
        };

        info!(
            "initialized {index_size} illusts ({pages} pages), {} not_founds, {} worker_to_downloads in {} ms",
            not_found.len(),
            worker_to_download.len(),
            t.elapsed().as_millis(),
        );
        Ok(Pvg {
            conf: config,
            index: TokioRwLock::new(nav),
            api: TokioRwLock::new(api),
            uid,
            pixiv: DownloadClient::new(),
            download_all_lock: TokioMutex::default(),
            not_found: Mutex::new(not_found),
            disk_lru: RwLock::new(lru),
            lru_limit,
            upscaler,
            update_lock: TokioMutex::default(),
            worker_to_download: TokioMutex::new(worker_to_download),
        })
    }

    pub fn is_worker(&self) -> bool {
        self.conf.worker_delay_secs > 0
    }

    async fn dump_cache(&self) -> serde_json::Result<String> {
        let api = self.api.read().await;
        let worker_to_download = self.worker_to_download.lock().await;
        let token = &api.state;
        let not_found = self.not_found.lock();
        info!(
            "cache: {} {}, {} not_founds, {} worker_to_downloads",
            token.user.id,
            token.user.name,
            not_found.len(),
            worker_to_download.len()
        );
        let cache = SavingCache {
            token,
            dims: vec![],
            not_found: &not_found,
            worker_to_download: &worker_to_download,
        };
        serde_json::to_string(&cache)
    }

    pub async fn save_cache(&self) -> Result<()> {
        let s = self.dump_cache().await?;
        let len = s.len();
        let file = &self.conf.cache_file;
        let tmp = file.with_extension("tmp");
        tokio::fs::write(&tmp, &s).await?;
        tokio::fs::rename(&tmp, &file).await?;
        info!("cache: written {} bytes into {}", len, file.display());
        Ok(())
    }

    async fn do_quick_update_with_page(&self, stage: usize, r: Vec<Value>) -> Result<bool> {
        let mut updated = false;
        let mut index = self.index.write().await;
        for illust in r {
            if index.stage(stage, illust)? {
                updated = true;
            }
        }
        Ok(!updated)
    }

    async fn auth(&self) -> Result<tokio::sync::RwLockReadGuard<'_, AuthedClient>> {
        {
            let mut api = self.api.write().await;
            api.ensure_authed().await?;
        }
        Ok(self.api.read().await)
    }

    async fn do_quick_update(&self, stage: usize, restrict: Restrict, pn_limit: u32) -> Result<()> {
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
        if self.do_quick_update_with_page(stage, r.illusts).await? {
            return Ok(());
        }
        while let Some(u) = r.next_url {
            pn += 1;
            if pn > pn_limit {
                break;
            }
            tokio::time::sleep(Duration::from_secs(3)).await;
            r = loop {
                match self.auth().await?.call_url(&u).await {
                    Ok(x) => {
                        break x;
                    }
                    Err(e) => {
                        // TODO: this is flawed
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
            if self.do_quick_update_with_page(stage, r.illusts).await? {
                break;
            }
        }
        Ok(())
    }

    async fn quick_update_atomic(&self) -> Result<RwLockWriteGuard<'_, IllustIndex>> {
        let _ = self.update_lock.try_lock()?;
        let empty = {
            let index = self.index.read().await;
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
            self.do_quick_update(0, Restrict::Private, limit),
            self.do_quick_update(1, Restrict::Public, limit),
        );
        let mut index = self.index.write().await;
        if let Err(e) = r {
            error!(
                "quick update failed ({} + {} rolled back): {:?}",
                index.rollback(1),
                index.rollback(0),
                e,
            );
            return Err(e);
        }
        Ok(index)
    }

    async fn commit_stage_worker(
        &self,
        index: &mut RwLockWriteGuard<'_, IllustIndex>,
        stage: usize,
        name: &'static str,
        res: &mut Vec<IllustId>,
    ) {
        {
            let it = index.peek(stage);
            let n = it.len();
            if n > 0 {
                res.extend(it.clone().map(|(id, _)| id));
                let it_new = it.clone().filter(|&(_, is_new)| is_new);
                let new = it_new.clone().count();
                if new > 0 {
                    let s_new = it_new.map(|(id, _)| id).join(", ");
                    let s_old = it
                        .filter(|&(_, is_new)| !is_new)
                        .map(|(id, _)| id)
                        .join(", ");
                    info!("{name}: {n} illusts ({new} new): {s_new} | {s_old}");
                } else if log::log_enabled!(log::Level::Debug) {
                    let s_all = it.map(|(id, _)| id).join(", ");
                    debug!("{name}: {n} illusts updated: {s_all}");
                }
            }
        }
        index.commit(stage).await;
    }

    pub async fn quick_update(&self) -> Result<(usize, usize)> {
        let (n_pri, n_pub) = {
            let mut index = self.quick_update_atomic().await?;
            // commit private first
            let n_pri = index.commit(0).await;
            let n_pub = index.commit(1).await;
            (n_pri, n_pub)
        };
        info!("quick updated {n_pub} + {n_pri} illusts");
        Ok((n_pri, n_pub))
    }

    async fn quick_update_worker(&self) -> Result<Vec<IllustId>> {
        let mut ids = Vec::new();
        {
            let mut index = self.quick_update_atomic().await?;
            self.commit_stage_worker(&mut index, 0, "private", &mut ids)
                .await;
            self.commit_stage_worker(&mut index, 1, "public", &mut ids)
                .await;
        }
        Ok(ids)
    }

    pub async fn get_source(&self, iid: IllustId, pn: PageNum) -> Option<(String, PathBuf)> {
        let index = self.index.read().await;
        let src = &index.get_page(iid, pn)?.source;
        let file = src.filename();
        let pile: &Path = file.as_ref();
        if self.not_found.lock().contains(pile) {
            return None;
        }
        if self.lru_limit.is_some() {
            self.disk_lru.write().promote(file);
        }
        let url = src.url();
        let path = self.page_path(pile);
        Some((url, path))
    }

    pub async fn upscale(&self, iid: IllustId, pn: PageNum, scale: u8) -> Result<PathBuf> {
        let file = {
            let index = self.index.read().await;
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
            error!("failed to remove {}: {e}", path.display());
        }
    }

    async fn disk_orphan(&self, file: &str) {
        let path = self.page_path(file);
        let dst = self.orphan_path(file);
        if let Err(e) = fs::rename(&path, &dst).await {
            error!("failed to orphan {}: {e}", path.display());
        }
    }

    fn disk_evict(&self, limit: u64) -> Option<String> {
        self.disk_lru.write().evict(limit)
    }

    async fn disk_evict_to(&self, limit: u64) {
        while let Some(file) = self.disk_evict(limit) {
            self.disk_remove(&file).await;
        }
    }

    async fn disk_evict_all(&self) {
        if let Some(limit) = self.lru_limit {
            self.disk_evict_to(limit).await;
        }
    }

    pub async fn enforce_cache_limit(&self) {
        if let Some(limit) = self.conf.cache_limit {
            self.disk_evict_to(limit).await;
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
                    warn!("{}: 404, memorized", path.display());
                    self.not_found
                        .lock()
                        .insert(path.file_name().unwrap().into());
                } else {
                    error!("{}: request failed: {e:?}", path.display());
                }
                return Err(e.into());
            }
            Ok(r) => r,
        };
        let size = remote.content_length();
        debug!(
            "{}: connection established in {:.3} secs, {:?} B",
            path.display(),
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
                        error!("{}: failed to write temp: {e}", path.display());
                        drop(perm);
                        tmp.rollback().await;
                        return;
                    }
                } else {
                    drop(perm);
                    match tmp.commit(&path, size).await {
                        Err(e) => error!("{}: failed to save: {e}", path.display()),
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
            error!("{}: remote error", path.display());
            drop(perm);
            tmp.rollback().await;
        });

        Ok((DownloadingStream { remote, tx, path }).stream())
    }

    async fn write_download_file(
        mut resp: pixiv::reqwest::Response,
        file: &mut DownloadingFile,
    ) -> Result<()> {
        while let Some(b) = resp.chunk().await? {
            file.write(&b).await?;
        }
        Ok(())
    }

    async fn do_download_file(&self, url: &str, path: &Path) -> Result<u64> {
        let perm = DOWNLOAD_SEMA.acquire().await?;
        let mut tmp = self.open_temp(path).await?;
        let r = match self.pixiv.download(url).await {
            Err(e) => {
                tmp.rollback().await;
                info!("{}: connection failed: {e:?}", path.display());
                return Err(e.into());
            }
            Ok(r) => r,
        };
        let size = r.content_length();

        tmp.start();
        let res = Self::write_download_file(r, &mut tmp).await;
        drop(perm);
        if let Err(e) = res {
            tmp.rollback().await;
            return Err(e);
        }
        tmp.commit(path, size).await
    }

    async fn download_file(&self, url: String, path: PathBuf) -> (PathBuf, Result<u64>) {
        let r = self.do_download_file(&url, &path).await;
        (path, r)
    }

    async fn make_download_queue(&self) -> Vec<(String, PathBuf)> {
        let index = self.index.read().await;
        let not_found = self.not_found.lock();
        let disk = self.disk_lru.read();
        let mut cnt_404 = 0;

        let r = index
            .iter()
            .flat_map(|illust| illust.pages.iter())
            .map(|page| (&page.source, page.source.filename()))
            .filter(|(_, file)| {
                if !disk.contains(file) {
                    let file: &Path = file.as_ref();
                    if not_found.contains(file) {
                        cnt_404 += 1;
                    } else {
                        return true;
                    }
                }
                false
            })
            .map(|(src, file)| (src.url(), self.page_path(file)))
            .collect();
        if cnt_404 > 0 {
            warn!("{cnt_404} pages skipped due to 404");
        }
        r
    }

    async fn make_download_queue_from(&self, ids: &[IllustId]) -> Vec<(String, PathBuf)> {
        let index = self.index.read().await;
        let not_found = self.not_found.lock();
        let disk = self.disk_lru.read();
        let mut cnt_404 = 0;
        let r = ids
            .iter()
            .flat_map(|iid| index.map[iid].pages.iter())
            .filter(|page| {
                let file = page.source.filename();
                if !disk.contains(file) {
                    let file: &Path = file.as_ref();
                    if not_found.contains(file) {
                        cnt_404 += 1;
                    } else {
                        return true;
                    }
                }
                false
            })
            .map(|page| (page.source.url(), self.page_path(page.source.filename())))
            .collect();
        if cnt_404 > 0 {
            warn!("{cnt_404} pages skipped due to 404");
        }
        r
    }

    pub async fn download_all(&self) -> Result<u32> {
        let _ = self.download_all_lock.try_lock()?;
        let q = self.make_download_queue().await;
        let (r, dirty_404) = self.do_download_all(q).await?;
        if dirty_404 {
            self.save_cache().await?;
        }
        Ok(r)
    }

    async fn download_all_worker(&self, ids: &[IllustId]) -> Result<(u32, bool)> {
        let _ = self.download_all_lock.try_lock()?;
        let q = self.make_download_queue_from(ids).await;

        // The caller is responsible for saving cache.
        self.do_download_all(q).await
    }

    async fn do_download_all(&self, q: Vec<(String, PathBuf)>) -> Result<(u32, bool)> {
        if self.lru_limit.is_some() {
            bail!("cache limit is set, refusing to download all");
        }
        if q.is_empty() {
            debug!("no pages to download");
            return Ok((0, false));
        }
        debug!("{} pages to download", q.len());
        let mut futs = q
            .into_iter()
            .map(|(url, path)| self.download_file(url, path))
            .collect::<FuturesUnordered<_>>();
        let n = futs.len();
        let mut cnt: u32 = 0;
        let mut cnt_fail: u32 = 0;
        let mut tot_size: u64 = 0;
        let mut the_404 = vec![];
        let t0 = Instant::now();
        while let Some((path, res)) = futs.next().await {
            cnt += 1;
            match res {
                Ok(size) => {
                    info!(
                        "{}/{}: downloaded {} ({} KiB)",
                        cnt,
                        n,
                        path.display(),
                        size >> 10
                    );
                    self.disk_lru
                        .write()
                        .insert(path.file_name().unwrap().to_str().unwrap().to_owned(), size);
                    tot_size += size;
                }
                Err(e) => {
                    cnt_fail += 1;
                    error!("{cnt}/{n}: download failed: {e}");
                    // handle 404 (and 500 for 85136899?)
                    if let Ok(pixiv::Error::Pixiv(404, _)) = e.downcast::<pixiv::Error>() {
                        warn!("{}: 404!", path.display());
                        the_404.push(path.file_name().unwrap().into());
                    }
                }
            }
        }
        let dt = t0.elapsed();
        drop(futs);
        let mut dirty_404 = false;
        if !the_404.is_empty() {
            let mut not_found = self.not_found.lock();
            let old_len = not_found.len();
            let cur_len = the_404.len();
            not_found.extend(the_404.into_iter());
            let new_len = not_found.len();
            if new_len > old_len {
                warn!("collected {} not_founds (+{})", cur_len, new_len - old_len);
                dirty_404 = true;
            }
        }
        if cnt_fail > 0 {
            bail!("failed to download {cnt_fail} pages out from {cnt}");
        }
        info!(
            "downloaded {cnt} pages ({:.3} MiB in {dt:.3?}, {:.3} KiB/s)",
            tot_size as f64 / 1024.0 / 1024.0,
            tot_size as f64 / (dt.as_secs_f64() * 1024.0),
        );
        Ok((cnt, dirty_404))
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
struct SelectedPage<'a>(Dimension, Dimension, &'a str, &'a str);

#[derive(Debug, Serialize)]
struct SelectedIllust<'a>(
    IllustId,
    &'a str,
    u32,
    &'a str,
    Vec<&'a str>,
    Vec<SelectedPage<'a>>,
    &'a str,
    u8,
);

#[derive(Debug, Serialize)]
struct SelectResponse<'a> {
    items: Vec<SelectedIllust<'a>>,
}

impl Pvg {
    pub async fn select(
        &self,
        mut filters: Vec<String>,
        ban_filters: Option<Vec<String>>,
    ) -> serde_json::Result<String> {
        if self.conf.safe_mode {
            filters.push("$s2".to_owned());
        }

        let now = Instant::now();
        let index = self.index.read().await;
        let r = index.select(
            &filters,
            if let Some(bans) = &ban_filters {
                bans
            } else {
                &[]
            },
        );

        let r = Self::_select(&index, r);

        let now2 = Instant::now();
        let dt = now2 - now;
        let n = r.len();
        let r = serde_json::to_string(&SelectResponse { items: r })?;
        let dt2 = now2.elapsed();
        info!(
            "{filters:?} - {:?}: {n} illusts ({:.1} KiB) in {dt:?} + {dt2:?}",
            if let Some(bans) = ban_filters.as_deref() {
                bans
            } else {
                &[]
            },
            r.len() as f32 / 1024.
        );
        Ok(r)
    }

    fn _select<'a>(
        index: &'a IllustIndex,
        r: impl DoubleEndedIterator<Item = &'a Illust>,
    ) -> Vec<SelectedIllust<'a>> {
        r.rev()
            .map(|illust| {
                let tags: Vec<&str> = index.tags(illust).collect();
                let mut curr_w = 0;
                let mut curr_h = 0;
                let pages: Vec<SelectedPage> = illust
                    .pages
                    .iter()
                    .map(|page| {
                        if let Some(d) = page.dimensions {
                            curr_w = d.0.get();
                            curr_h = d.1.get();
                        }
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
                    i.user_id,
                    index.user_name(illust),
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
    async fn orphan(&self) -> Vec<String> {
        let index = self.index.read().await;
        let s: HashSet<_> = index
            .iter()
            .flat_map(|illust| illust.pages.iter())
            .map(|page| page.source.filename())
            .collect();

        let mut lru = self.disk_lru.write();
        lru.filter(|file| s.contains(file))
    }

    pub async fn remove_orphans(&self) -> usize {
        let a = self.orphan().await;
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
        let a = self.orphan().await;
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

        // What were we forgetting to download?
        let mut wal = self.worker_to_download.lock().await;
        let has_wal = !wal.is_empty();

        if has_wal {
            let old_n = wal.len();
            wal.extend(ids);
            warn!(
                "worker: remaining {old_n} illusts to download (total {})",
                wal.len()
            );
        } else {
            *wal = ids;
        }

        if wal.is_empty() {
            // Nothing to do.
            return Ok(());
        }

        // Dropping the lock, prevent `save_cache()` from being blocked.
        let todo = wal.clone();
        drop(wal);

        let dirty_cache = match self.download_all_worker(&todo).await {
            Ok((n, dirty_404)) => {
                debug!("worker: downloaded {n} illusts");
                dirty_404 || has_wal
            }
            Err(e) => {
                error!("download_all_worker: {e}");
                has_wal
            }
        };

        // Clear them even if failed to download, as we never retry for now.
        // Some illusts just keep failing with 404/500.
        {
            let mut wal = self.worker_to_download.lock().await;
            // We are the only writer.
            assert_eq!(*wal, todo);
            wal.clear();
        }

        if dirty_cache {
            self.save_cache().await?;
        }

        self.move_orphans().await;
        Ok(())
    }

    // As a worker, we simply poll updates and download them.
    // Disk states should be never checked in worker mode, as the user can move any downloaded stuff away.
    pub async fn worker_start(&self) {
        if self.conf.worker_delay_secs > 0 {
            let d = Duration::from_secs(u64::from(self.conf.worker_delay_secs));
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
