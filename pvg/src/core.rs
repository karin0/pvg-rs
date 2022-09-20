use crate::config::{read_config, Config};
use crate::model::{DimCache, Dimensions, IllustIndex};
use actix_web::web::Bytes;
use anyhow::{bail, Context, Result};
use fs2::FileExt;
use futures::stream::FuturesUnordered;
use futures::{stream, Stream, StreamExt};
use image::GenericImageView;
use parking_lot::{Mutex, RwLock};
use pixiv::aapi::BookmarkRestrict;
use pixiv::client::{AuthedClient, AuthedState};
use pixiv::download::DownloadClient;
use pixiv::{IllustId, PageNum};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value};
use std::collections::HashSet;
use std::ffi::OsString;
use std::io;
use std::io::Read;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;
use tokio::time::Instant;

#[derive(Deserialize, Default)]
struct LoadedCache {
    token: Option<AuthedState>,
    dims: Option<Vec<(IllustId, Vec<u32>)>>,
    #[serde(default)]
    not_found: HashSet<OsString>,
}

#[derive(Serialize)]
struct SavingCache<'a> {
    token: &'a AuthedState,
    dims: Vec<(IllustId, Vec<u32>)>,
    not_found: &'a HashSet<OsString>,
}

#[derive(Debug)]
pub struct Pvg {
    conf: Config,
    #[allow(dead_code)]
    lock: std::fs::File,
    index: RwLock<IllustIndex>,
    api: tokio::sync::RwLock<AuthedClient>,
    pixiv: DownloadClient,
    uid: String,
    disk: tokio::sync::Mutex<()>,
    download_all_lock: tokio::sync::Mutex<()>,
    not_found: Mutex<HashSet<OsString>>,
    db_init_size: usize,
}

#[derive(Deserialize, Debug)]
pub struct BookmarkPage {
    illusts: Vec<Map<String, Value>>,
    next_url: Option<String>,
}

#[derive(Debug)]
struct DownloadingFile {
    path: PathBuf,
    file: fs::File,
}

impl DownloadingFile {
    pub async fn new(path: PathBuf) -> Result<Self> {
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .await?;
        Ok(Self { path, file })
    }

    pub async fn write(&mut self, b: &Bytes) -> io::Result<()> {
        self.file.write_all(b).await
    }

    pub async fn commit(self, path: &Path) -> io::Result<()> {
        drop(self.file);
        if let Err(e) = fs::rename(&self.path, path).await {
            error!("{:?}: COMMIT FAILED: {}", self.path, e);
            Err(e)
        } else {
            Ok(())
        }
    }

    pub async fn rollback(self) {
        drop(self.file);
        if let Err(e) = fs::remove_file(&self.path).await {
            error!("{:?}: ROLLBACK FAILED: {}", self.path, e);
        }
    }
}

impl Pvg {
    pub async fn new() -> Result<Self> {
        let t = Instant::now();
        let config = read_config()?;
        info!("config: {:?}", config);
        // XXX: using sync files since fs2 doesn't support async
        let (mut nav, lock) = match std::fs::File::open(&config.db_file) {
            Ok(mut db) => {
                db.try_lock_exclusive()?;
                let mut s = String::new();
                db.read_to_string(&mut s)?;
                info!("read {} bytes from {:?}", s.len(), db);
                let nav = IllustIndex::parse(s)?;
                (nav, db)
            }
            Err(e) => {
                if e.kind() == std::io::ErrorKind::NotFound {
                    info!("creating new db file");
                    let lock = std::fs::OpenOptions::new()
                        .create_new(true)
                        .open(&config.db_file)?;
                    lock.try_lock_exclusive()?;
                    (IllustIndex::default(), lock)
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
                if e.kind() == std::io::ErrorKind::NotFound {
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

        let db_init_size = nav.len();
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
                error!(
                    "DB SIZE CHANGED WITHOUT DIRTY FLAG! {} -> {}",
                    self.db_init_size, n
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

    fn _quick_update_with_page(&self, r: Vec<Map<String, Value>>) -> Result<bool> {
        let mut updated = false;
        let mut index = self.index.write();
        for illust in r {
            if index.stage(illust)? {
                updated = true;
            }
        }
        Ok(!updated)
    }

    async fn _quick_update(&self) -> Result<()> {
        let mut api = self.api.try_write()?;
        api.ensure_authed().await?;
        drop(api);
        let api = self.api.read().await;
        let mut r: BookmarkPage = api
            .user_bookmarks_illust(&self.uid, BookmarkRestrict::Private)
            .await?;
        let mut pn = 1;
        info!("page {}: {} illusts", pn, r.illusts.len());
        if self._quick_update_with_page(r.illusts)? {
            return Ok(());
        }
        while let Some(u) = r.next_url {
            r = api.call_url(&u).await?;
            pn += 1;
            info!("page {}: {} illusts", pn, r.illusts.len());
            if self._quick_update_with_page(r.illusts)? {
                break;
            }
        }
        Ok(())
    }

    pub async fn quick_update(&self) -> Result<()> {
        if let Err(e) = self._quick_update().await {
            let n = self.index.write().rollback();
            error!("quick update failed ({} rolled back): {}", n, e);
        }
        info!("quick updated {} illusts", self.index.write().commit());
        Ok(())
    }

    pub fn get_source(&self, iid: IllustId, pn: PageNum) -> Option<(String, PathBuf)> {
        let index = self.index.read();
        let pn: usize = pn.try_into().ok()?;
        let src = &index.map.get(&iid)?.pages.get(pn)?.source;
        let url = src.url.clone();
        let path = self.conf.pix_dir.join(src.filename());
        Some((url, path))
    }

    async fn open_temp(&self, path: &Path) -> Result<DownloadingFile> {
        let tmp_path = self.conf.tmp_dir.join(path.file_name().unwrap());
        // TODO: this fails if a file is requested twice before it's downloaded, do some waiting.
        // FIXME: this always fails if an old temp file is left behind.
        DownloadingFile::new(tmp_path).await
    }

    pub async fn download(
        &self,
        src: &str,
        path: PathBuf,
    ) -> Result<impl Stream<Item = pixiv::reqwest::Result<Bytes>>> {
        info!("downloading {}", src);
        let mut tmp = self.open_temp(&path).await?;
        let (tx, mut rx) = mpsc::unbounded_channel::<Option<Bytes>>();

        let npath = path.clone();
        tokio::spawn(async move {
            let path = npath;
            while let Some(msg) = rx.recv().await {
                if let Some(b) = msg {
                    if let Err(e) = tmp.write(&b).await {
                        error!("{:?}: failed to write temp: {}", path, e);
                        tmp.rollback().await;
                        return;
                    }
                } else {
                    error!("{:?}: remote error", path);
                    tmp.rollback().await;
                    return;
                }
            }
            info!("{:?}: committing", path);
            if let Err(e) = tmp.commit(&path).await {
                error!("{:?}: failed to save: {}", path, e);
            }
        });

        let remote = self.pixiv.download(src).await?.bytes_stream();
        Ok(stream::unfold(
            (remote, tx, path),
            |(mut remote, tx, path)| async move {
                match remote.next().await {
                    Some(Ok(b)) => {
                        if let Err(e) = tx.send(Some(b.clone())) {
                            error!("{:?}: send error: {}", path, e);
                        }
                        Some((Ok(b), (remote, tx, path)))
                    }
                    Some(Err(e)) => {
                        error!("{:?}: remote streaming failed: {}", path, e);
                        if let Err(e) = tx.send(None) {
                            error!("{:?}: none send error: {}", path, e);
                        }
                        Some((Err(e), (remote, tx, path)))
                    }
                    None => {
                        info!("{:?}: remote streaming done", path);
                        drop(tx);
                        None
                    }
                }
            },
        ))
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

    async fn _downloader_inner(
        &self,
        url: &str,
        path: &Path,
        sema: &tokio::sync::Semaphore,
    ) -> Result<()> {
        let perm = sema.acquire().await?;
        let r = self.pixiv.download(url).await?;

        let mut tmp = self.open_temp(path).await?;
        let res = Self::_download_file(r, &mut tmp).await;
        drop(perm);
        if let Err(e) = res {
            tmp.rollback().await;
            return Err(e);
        }
        tmp.commit(path).await?;
        Ok(())
    }

    async fn _downloader(
        &self,
        url: String,
        path: PathBuf,
        sema: &tokio::sync::Semaphore,
    ) -> (PathBuf, Result<()>) {
        let r = self._downloader_inner(&url, &path, sema).await;
        (path, r)
    }

    fn _make_download_queue(&self) -> Vec<(String, PathBuf)> {
        let not_found = self.not_found.lock();
        self.index
            .read()
            .iter()
            .flat_map(|illust| illust.pages.iter())
            .map(|page| (&page.source, self.conf.pix_dir.join(page.source.filename())))
            .filter(|(_, path)| {
                if !path.exists() {
                    if !not_found.contains(path.file_name().unwrap()) {
                        return true;
                    } else {
                        warn!("{:?}: skipping due to 404", path);
                    }
                }
                false
            })
            .map(|(src, path)| (src.url.clone(), path))
            .collect()
    }

    pub async fn download_all(&self) -> Result<()> {
        // FIXME: generate the whole queue in sync for now to avoid the use of async locks.
        let _ = self.download_all_lock.try_lock()?;
        let t = Instant::now();
        let q = self._make_download_queue();
        warn!("download_all: blocked for {} ms", t.elapsed().as_millis());
        info!("{} pages to download", q.len());
        let sema = tokio::sync::Semaphore::new(5);
        let mut futs = q
            .into_iter()
            .map(|(url, path)| self._downloader(url, path, &sema))
            .collect::<FuturesUnordered<_>>();
        let n = futs.len();
        let mut cnt = 0;
        let mut cnt_fail = 0;
        let mut the_404 = vec![];
        while let Some((path, res)) = futs.next().await {
            cnt += 1;
            match res {
                Ok(path) => info!("{}/{}: downloaded {:?}", cnt, n, path),
                Err(e) => {
                    cnt_fail += 1;
                    error!("{}/{}: download failed: {}", cnt, n, e);
                    // handle 404 (and 500 for 85136899?)
                    if let Ok(pixiv::Error::Pixiv(404, _)) = e.downcast::<pixiv::Error>() {
                        warn!("{:?}: 404!", path);
                        the_404.push(path.file_name().unwrap().to_owned());
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

    fn _make_measure_queue(&self) -> (Vec<(IllustId, Vec<(PageNum, String)>)>, usize) {
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

    fn _measure(path: &Path) -> Result<Dimensions> {
        let img = image::open(path)?;
        let (w, h) = img.dimensions();
        Ok(Dimensions(w.try_into()?, h.try_into()?))
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
                    let r = Self::_measure(&file);
                    if let Err(e) = tx.send((iid, pn, r)) {
                        error!("{:?}: send error: {}", file, e);
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

    pub fn select(&self, filters: &[String]) -> Vec<Vec<Value>> {
        // TODO: do this all sync can block for long.
        let index = self.index.read();
        let now = Instant::now();
        let r: Vec<Vec<Value>> = index
            .select(filters)
            .rev()
            .flat_map(|illust| {
                illust.pages.iter().enumerate().flat_map(|(i, page)| {
                    let dims;
                    if let Some(d) = page.dimensions {
                        dims = d;
                    } else {
                        warn!("skipping unmeasured page {}", page.source.filename());
                        return None;
                    }
                    Some(vec![
                        Value::Number(Number::from(illust.data.id)),
                        Value::Number(Number::from(i)),
                        Value::String("img".to_string()),
                        Value::Number(Number::from(dims.0.get())),
                        Value::Number(Number::from(dims.1.get())),
                        Value::String(illust.data.title.clone()),
                        Value::String(illust.data.user.name.clone()),
                        Value::Number(Number::from(illust.data.user.id)),
                        Value::Array(
                            illust
                                .data
                                .tags
                                .iter()
                                .map(|t| Value::String(t.name.clone()))
                                .collect(),
                        ),
                        Value::String(page.source.filename().to_string()),
                    ])
                })
            })
            .collect();
        let t = now.elapsed();
        drop(index);
        info!("{:?} -> {} results, {} ms", filters, r.len(), t.as_millis());
        r
    }
}
