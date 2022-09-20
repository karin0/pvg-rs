use anyhow::Result;
use serde::Deserialize;
use serde_json::from_str;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Deserialize, Debug)]
struct ConfigFile {
    username: String,
    refresh_token: String,
    home: Option<PathBuf>,
    proxy: Option<String>,
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

    Ok(Config {
        username: config.username,
        refresh_token: config.refresh_token,
        proxy: config.proxy,
        pix_dir: at_dir("pix"),
        tmp_dir: at_dir("tmp"),
        db_file: at("fav.json"),
        cache_file: at("cache.json"),
    })
}
