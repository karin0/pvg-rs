mod config;
mod model;
mod pixiv;

use crate::config::{read_config, Config};
use crate::model::IllustIndex;
use crate::pixiv::{IllustId, PageNum};
use actix_cors::Cors;
use actix_web::{get, post, web, App, HttpServer, Responder};
use anyhow::Result;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};
use std::io;
use std::path::PathBuf;
use time::Instant;

#[macro_use]
extern crate log;
extern crate core;

#[derive(Debug)]
struct Pvg {
    conf: Config,
    index: RwLock<IllustIndex>,
}

#[derive(Debug, Serialize)]
struct SelectResponse {
    items: Vec<Vec<Value>>,
}

impl Pvg {
    fn get_file(&self, iid: IllustId, pn: PageNum) -> Option<PathBuf> {
        let index = self.index.read();
        let file = &index.map.get(&iid)?.pages.get(pn as usize)?.filename;
        let res = self.conf.pix_dir.join(file);
        drop(index);
        Some(res)
    }

    fn select(&self, filters: &[String]) -> SelectResponse {
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
        info!(
            "{:?} -> {} results, {} ms",
            filters,
            r.len(),
            t.whole_milliseconds(),
        );
        SelectResponse { items: r }
    }
}

#[get("/img/{iid}/{pn}")]
async fn image(app: web::Data<Pvg>, path: web::Path<(IllustId, PageNum)>) -> impl Responder {
    let (iid, pn) = path.into_inner();
    match app.get_file(iid, pn) {
        Some(file) => actix_files::NamedFile::open_async(file).await,
        None => Err(io::Error::new(io::ErrorKind::NotFound, "Not found")),
    }
}

#[derive(Deserialize, Debug)]
struct SelectPayload {
    filters: Vec<String>,
}

#[post("/select")]
async fn select(app: web::Data<Pvg>, filters: web::Json<SelectPayload>) -> impl Responder {
    web::Json(app.select(&filters.filters))
}

#[actix_web::main] // or #[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    pretty_env_logger::init_timed();

    let config = read_config().await?;
    info!("config: {:?}", config);
    let nav = IllustIndex::new(&config.db_file).await?;
    info!("index got {} illusts", nav.map.len());

    let data = Pvg {
        conf: config,
        index: RwLock::new(nav),
    };
    let data = web::Data::new(data);
    HttpServer::new(move || {
        App::new()
            .wrap(Cors::permissive())
            .app_data(data.clone())
            .service(image)
            .service(select)
    })
    .bind(("127.0.0.1", 5678))?
    .run()
    .await
    .unwrap();

    Ok(())
}
