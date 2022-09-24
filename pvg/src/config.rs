use anyhow::Result;
use serde::Deserialize;
use serde_json::from_str;
use std::fs;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};

#[derive(Deserialize, Debug)]
struct ConfigFile {
    username: String,
    refresh_token: String,
    home: Option<PathBuf>,
    proxy: Option<String>,
    static_dir: Option<PathBuf>,
    host: Option<IpAddr>,
    port: Option<u16>,
    cache_limit: Option<u64>,
    // pix_dir: String,
    // tmp_dir: String,
}

#[derive(Debug)]
pub struct Config {
    pub username: String,
    pub refresh_token: String,
    pub proxy: Option<String>,
    pub pix_dir: PathBuf,
    pub tmp_dir: PathBuf,
    pub db_file: PathBuf,
    pub cache_file: PathBuf,
    pub static_dir: PathBuf,
    pub addr: SocketAddr,
    pub cache_limit: Option<u64>,
}

fn ensure_dir(dir: &Path) {
    match fs::metadata(dir) {
        Ok(meta) => {
            if !meta.is_dir() {
                panic!("{} is not a directory", dir.display());
            }
        }
        Err(_) => {
            fs::create_dir(dir).unwrap();
        }
    };
}

fn ensure_empty_dir(dir: PathBuf) -> PathBuf {
    if let Ok(meta) = fs::metadata(&dir) {
        if !meta.is_dir() {
            panic!("{} is not a directory", dir.display());
        }
        fs::remove_dir_all(&dir).unwrap();
    };
    fs::create_dir(&dir).unwrap();
    dir
}

pub fn read_config() -> Result<Config> {
    let config = fs::read_to_string("config.json")?;
    let config: ConfigFile = from_str(&config)?;

    let home = &config.home;
    let at = |f: &str| match home {
        Some(home) => home.join(f),
        _ => f.into(),
    };
    let at_dir = |f: &str| {
        let r = match home {
            Some(home) => home.join(f),
            _ => f.into(),
        };
        ensure_dir(&r);
        r
    };

    let static_dir = config.static_dir.unwrap_or_else(|| at("static"));

    let host = config.host.unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST));
    let port = config.port.unwrap_or(5678);
    let addr = SocketAddr::new(host, port);

    Ok(Config {
        username: config.username,
        refresh_token: config.refresh_token,
        proxy: config.proxy,
        pix_dir: at_dir("pix"),
        tmp_dir: ensure_empty_dir(at("tmp")),
        db_file: at("fav.json"),
        cache_file: at("cache.json"),
        static_dir,
        addr,
        cache_limit: config.cache_limit,
    })
}
