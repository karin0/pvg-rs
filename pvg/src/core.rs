use crate::config::{read_config, Config};
use crate::model::IllustIndex;
use anyhow::Result;
use fs2::FileExt;
use parking_lot::RwLock;
use pixiv::aapi::BookmarkRestrict;
use pixiv::client::AuthedClient;
use pixiv::{IllustId, PageNum};
use serde::Deserialize;
use serde_json::{Map, Number, Value};
use std::io::Read;
use std::path::PathBuf;
use tokio::time::Instant;

#[derive(Debug)]
pub struct Pvg {
    conf: Config,
    lock: std::fs::File,
    index: RwLock<IllustIndex>,
    api: RwLock<AuthedClient>,
    uid: String,
    disk: tokio::sync::Mutex<()>,
}

#[derive(Deserialize, Debug)]
pub struct BookmarkPage {
    illusts: Vec<Map<String, Value>>,
    next_url: Option<String>,
}

impl Pvg {
    pub async fn new() -> Result<Self> {
        let t = Instant::now();
        let config = read_config().await?;
        info!("config: {:?}", config);
        // XXX: using sync files since fs2 doesn't support async
        let (nav, lock) = match std::fs::File::open(&config.db_file) {
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
        let api = AuthedClient::new(&config.refresh_token).await?;
        info!("api: {} {}", api.state.user.name, api.state.user.id);
        let uid = api.state.user.id.to_string();

        Ok(Pvg {
            conf: config,
            lock,
            index: RwLock::new(nav),
            api: RwLock::new(api),
            disk: tokio::sync::Mutex::new(()),
            uid,
        })
    }

    pub async fn dump(&self) -> Result<()> {
        let index = self.index.read();
        let s = index.dump()?;
        drop(index);
        let _ = self.disk.lock().await;
        info!("writing {} bytes", s.len());
        tokio::fs::write(&self.conf.db_file, s).await?;
        info!("written into {:?}", self.conf.db_file);
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
        let api = self.api.read();
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

    pub fn get_file(&self, iid: IllustId, pn: PageNum) -> Option<PathBuf> {
        let index = self.index.read();
        let file = &index.map.get(&iid)?.pages.get(pn as usize)?.filename;
        let res = self.conf.pix_dir.join(file);
        drop(index);
        Some(res)
    }

    pub fn select(&self, filters: &[String]) -> Vec<Vec<Value>> {
        let index = self.index.read();
        let now = Instant::now();
        let r: Vec<Vec<Value>> = index
            .select(filters)
            .flat_map(|illust| {
                illust.pages.iter().enumerate().map(|(i, page)| {
                    vec![
                        Value::Number(Number::from(illust.data.id)),
                        Value::Number(Number::from(i)),
                        Value::String("img".to_string()),
                        Value::Number(Number::from(page.width)),
                        Value::Number(Number::from(page.height)),
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
                        Value::String(page.filename.clone()),
                    ]
                })
            })
            .collect();
        let t = now.elapsed();
        drop(index);
        info!("{:?} -> {} results, {} ms", filters, r.len(), t.as_millis());
        r
    }
}