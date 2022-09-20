use crate::config::{read_config, Config};
use crate::model::{DimCache, IllustIndex};
use actix_web::web::Bytes;
use anyhow::Result;
use fs2::FileExt;
use futures::{stream, Stream, StreamExt};
use parking_lot::RwLock;
use pixiv::aapi::BookmarkRestrict;
use pixiv::client::{AuthedClient, AuthedState};
use pixiv::download::DownloadClient;
use pixiv::{IllustId, PageNum};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value};
use std::io;
use std::io::Read;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;
use tokio::time::Instant;

#[derive(Deserialize)]
struct LoadedCache {
    token: Option<AuthedState>,
    dims: Option<Vec<(IllustId, Vec<u32>)>>,
}

#[derive(Serialize)]
struct SavingCache<'a> {
    token: &'a AuthedState,
    dims: Vec<(IllustId, Vec<u32>)>,
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
        fs::rename(&self.path, path).await
    }

    pub async fn rollback(self) {
        drop(self.file);
        if let Err(e) = fs::remove_file(&self.path).await {
            error!("{:?}: failed to remove failed temp: {}", self.path, e);
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
                    AuthedClient::new(&config.refresh_token).await?
                } else {
                    return Err(e.into());
                }
            }
        };

        info!("api: {} {}", api.state.user.name, api.state.user.id);
        let uid = api.state.user.id.to_string();

        Ok(Pvg {
            conf: config,
            lock,
            index: RwLock::new(nav),
            api: tokio::sync::RwLock::new(api),
            disk: tokio::sync::Mutex::new(()),
            uid,
            pixiv: DownloadClient::new(),
        })
    }

    fn dump_index(&self) -> Result<(Vec<u8>, DimCache)> {
        let index = self.index.read();
        let s = index.dump()?;
        let dims = index.dump_dims_cache();
        Ok((s, dims))
    }

    pub async fn dump(&self) -> Result<()> {
        let (s, dims) = self.dump_index()?;
        let _ = self.disk.lock().await;
        fs::rename(&self.conf.db_file, &self.conf.db_file.with_extension("bak")).await?;
        info!("writing {} bytes", s.len());
        fs::write(&self.conf.db_file, s).await?;
        info!("written into {:?}", self.conf.db_file);
        let api = self.api.read().await;
        info!("cache has {} dims", dims.len());
        let cache = SavingCache {
            token: &api.state,
            dims,
        };
        let s = serde_json::to_string(&cache)?;
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

    pub async fn download(
        &self,
        src: &str,
        path: PathBuf,
    ) -> Result<impl Stream<Item = pixiv::reqwest::Result<Bytes>>> {
        info!("downloading {}", src);
        let tmp_path = self.conf.tmp_dir.join(path.file_name().unwrap());
        // TODO: this fails if a file is requested twice before it's downloaded, do some waiting.
        // FIXME: this always fails if an old temp file is left behind.
        let mut tmp = DownloadingFile::new(tmp_path).await?;
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
