mod config;
mod core;
mod model;

use crate::core::Pvg;
use actix_cors::Cors;
use actix_files::NamedFile;
use actix_web::http::StatusCode;
use actix_web::{
    get, post, web, App, Either, HttpResponse, HttpResponseBuilder, HttpServer, Responder,
};
use anyhow::Result;
use pixiv::{IllustId, PageNum};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Debug;
use std::io;
use tokio::sync::oneshot;
use tokio::time::Instant;

#[macro_use]
extern crate log;

#[get("/img/{iid}/{pn}")]
async fn image(
    app: web::Data<Pvg>,
    path: web::Path<(IllustId, PageNum)>,
) -> io::Result<Either<NamedFile, HttpResponse>> {
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
    web::Json(SelectResponse {
        items: app.select(&filters.filters),
    })
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
            .service(quick_update)
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
