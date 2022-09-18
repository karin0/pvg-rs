use crate::config::{read_config, Config};
use crate::model::IllustIndex;
use anyhow::Result;
use parking_lot::RwLock;
use pixiv::client::AuthedClient;
use pixiv::{IllustId, PageNum};
use serde_json::{Number, Value};
use std::path::PathBuf;
use tokio::time::Instant;

#[derive(Debug)]
pub struct Pvg {
    conf: Config,
    index: RwLock<IllustIndex>,
    api: RwLock<AuthedClient>,
}

impl Pvg {
    pub async fn new() -> Result<Self> {
        let config = read_config().await?;
        info!("config: {:?}", config);
        let nav = IllustIndex::new(&config.db_file).await?;
        info!("index got {} illusts", nav.map.len());
        let api = AuthedClient::new(&config.refresh_token).await?;

        Ok(Pvg {
            conf: config,
            index: RwLock::new(nav),
            api: RwLock::new(api),
        })
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
