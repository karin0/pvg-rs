use crate::PageNum;
use anyhow::{bail, Result};
use serde::Deserialize;
use serde_json::from_str;
use std::io;
use std::path::{Path, PathBuf};
use tokio::fs;

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
}

async fn ensure_dir(dir: &Path) -> Result<()> {
    match fs::metadata(dir).await {
        Ok(meta) => {
            if !meta.is_dir() {
                // TODO: use io::Error: NotADirectory (which is nightly for now)
                // return io::Error::new(io::ErrorKind::NotADirectory, dir.to_owned()).into();
                bail!("{} is not a directory", dir.display());
            }
        }
        Err(_) => {
            fs::create_dir(dir).await.unwrap();
        }
    };
    Ok(())
}

pub async fn read_config() -> Result<Config> {
    let config = fs::read_to_string("config.json").await?;
    let config: ConfigFile = from_str(&config)?;
    let (pix_dir, tmp_dir, db_file) = match config.home {
        Some(s) => (s.join("pix"), s.join("tmp"), s.join("fav.json")),
        _ => ("pix".into(), "tmp".into(), "fav.json".into()),
    };
    ensure_dir(&pix_dir).await?;
    ensure_dir(&tmp_dir).await?;
    Ok(Config {
        username: config.username,
        refresh_token: config.refresh_token,
        proxy: config.proxy,
        pix_dir,
        tmp_dir,
        db_file,
    })
}
