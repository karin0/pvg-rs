mod config;
mod core;
mod disk_lru;
mod model;
mod util;

use crate::core::Pvg;
use actix_cors::Cors;
use actix_files::NamedFile;
use actix_web::http::{header::ContentType, StatusCode};
use actix_web::{
    get, post, web, App, Either, HttpRequest, HttpResponse, HttpResponseBuilder, HttpServer,
    Responder,
};
use anyhow::Result;
use pixiv::{IllustId, PageNum};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Debug;
use std::io;
use std::path::Path;
use tokio::sync::oneshot;
use tokio::time::Instant;

#[macro_use]
extern crate log;

#[get("/img/{iid}/{pn}")]
async fn image(
    app: web::Data<Pvg>,
    path: web::Path<(IllustId, PageNum)>,
) -> io::Result<Either<NamedFile, HttpResponse>> {
    let app = app.into_inner();
    let (iid, pn) = path.into_inner();
    match app.get_source(iid, pn) {
        Some((src, path)) => match NamedFile::open_async(&path).await {
            Ok(f) => Ok(Either::Left(f)),
            Err(e) => {
                if e.kind() == io::ErrorKind::NotFound {
                    // info!("refusing to download {:?}", path);
                    // return Err(io::Error::new(io::ErrorKind::Other, "bye"));
                    let resp = app.download(&src, path).await.map_err(mapper)?;
                    Ok(Either::Right(
                        HttpResponseBuilder::new(StatusCode::OK).streaming(resp),
                    ))
                } else {
                    Err(e)
                }
            }
        },
        _ => Err(io::Error::new(io::ErrorKind::NotFound, "Not found")),
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
    let r = app.select(&filters.filters).map_err(mapper)?;
    io::Result::Ok(HttpResponse::Ok().content_type(ContentType::json()).body(r))
}

fn mapper<T: Into<anyhow::Error>>(e: T) -> io::Error {
    let e = e.into();
    error!("mapper: {:?}", e);
    io::Error::new(io::ErrorKind::Other, e.to_string())
}

#[get("/action/qupd")]
async fn quick_update(app: web::Data<Pvg>) -> io::Result<&'static str> {
    app.quick_update().await.map_err(mapper)?;
    Ok("ok")
}

#[get("/action/download")]
async fn download_all(app: web::Data<Pvg>) -> io::Result<&'static str> {
    app.download_all().await.map_err(mapper)?;
    Ok("ok")
}

#[get("/action/measure")]
async fn measure_all(app: web::Data<Pvg>) -> io::Result<&'static str> {
    app.measure_all().await.map_err(mapper)?;
    Ok("ok")
}

#[get("/test")]
async fn test(app: web::Data<Pvg>) -> impl Responder {
    let t = Instant::now();
    app.dump().await.unwrap();
    info!("dump: {} ms", t.elapsed().as_millis());
    "ok"
}

#[get("/")]
async fn index(req: HttpRequest) -> impl Responder {
    let index: &'static Path = *req.app_data().unwrap();
    NamedFile::open_async(index).await
}

#[actix_web::main] // or #[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    pretty_env_logger::init_timed();

    let core = Pvg::new().await?;
    let static_dir: &'static Path = Box::leak(core.conf.static_dir.clone().into_boxed_path());
    let static_index: &'static Path = Box::leak(static_dir.join("index.html").into_boxed_path());
    let addr = core.conf.addr;
    let data = web::Data::new(core);
    let pvg = data.clone();
    let server = HttpServer::new(move || {
        let statics = actix_files::Files::new("/s", static_dir);
        App::new()
            .wrap(Cors::permissive())
            .app_data(data.clone())
            .app_data(static_index)
            .service(statics)
            .service(image)
            .service(select)
            .service(quick_update)
            .service(download_all)
            .service(measure_all)
            .service(test)
            .service(index)
    })
    .bind(addr)?
    .disable_signals()
    .run();
    info!("listening on {}", addr);
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
