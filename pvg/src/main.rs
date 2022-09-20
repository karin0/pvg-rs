mod config;
mod core;
mod model;

use crate::core::Pvg;
use actix_cors::Cors;
use actix_web::{get, post, web, App, HttpServer, Responder};
use anyhow::Result;
use pixiv::{IllustId, PageNum};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io;
use tokio::sync::oneshot;
use tokio::time::Instant;

#[macro_use]
extern crate log;

#[get("/img/{iid}/{pn}")]
async fn image(app: web::Data<Pvg>, path: web::Path<(IllustId, PageNum)>) -> impl Responder {
    let (iid, pn) = path.into_inner();
    match app.get_file(iid, pn) {
        Some(file) => actix_files::NamedFile::open_async(file).await,
        None => Err(io::Error::new(io::ErrorKind::NotFound, "Not found")),
    }
}

#[derive(Debug, Serialize)]
struct SelectResponse {
    items: Vec<Vec<Value>>,
}

#[derive(Deserialize, Debug)]
struct SelectPayload {
    filters: Vec<String>,
}

#[post("/select")]
async fn select(app: web::Data<Pvg>, filters: web::Json<SelectPayload>) -> impl Responder {
    web::Json(SelectResponse {
        items: app.select(&filters.filters),
    })
}

#[get("/action/qupd")]
async fn quick_update(app: web::Data<Pvg>) -> impl Responder {
    "ok"
}

#[get("/test")]
async fn test(app: web::Data<Pvg>) -> impl Responder {
    let t = Instant::now();
    app.dump().await.unwrap();
    info!("dump: {} ms", t.elapsed().as_millis());
    "ok"
}

#[actix_web::main] // or #[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    pretty_env_logger::init_timed();

    let data = web::Data::new(Pvg::new().await?);
    let pvg = data.clone();
    let server = HttpServer::new(move || {
        App::new()
            .wrap(Cors::permissive())
            .app_data(data.clone())
            .service(image)
            .service(select)
            .service(test)
    })
    .bind(("127.0.0.1", 5678))?
    .disable_signals()
    .run();
    let handle = server.handle();

    let (tx, mut rx) = oneshot::channel();
    let mut tx = Some(tx);
    ctrlc::set_handler(move || match tx.take() {
        Some(tx) => {
            if tx.send(()).is_err() {
                error!("failed to invoke shutdown");
            } else {
                warn!("shutting down");
            }
        }
        None => {
            warn!("is shutting down");
        }
    })?;

    tokio::select! {
        _ = server => {
            error!("server terminated unexpectedly");
        },
        _ = &mut rx => {}
    }
    pvg.dump().await?;
    info!("shutting down server");
    handle.stop(true).await;
    Ok(())
}