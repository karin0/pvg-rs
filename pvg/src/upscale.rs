use crate::config::Config;
use anyhow::{bail, Result};
use std::io;
use std::path::{Path, PathBuf};
use tokio::fs::canonicalize;
use tokio::process::Command;

#[derive(Debug)]
pub struct Upscaler {
    bin: PathBuf,
    cwd: PathBuf,
    abs_src_dir: PathBuf,
    abs_dst_dir: PathBuf,
}

impl Upscaler {
    pub async fn new(mut bin: PathBuf, config: &Config) -> io::Result<Self> {
        let abs = canonicalize(&bin).await?;
        bin.pop();
        Ok(Self {
            bin: abs,
            cwd: bin,
            abs_src_dir: canonicalize(&config.pix_dir).await?,
            abs_dst_dir: canonicalize(&config.upscale_dir).await?,
        })
    }

    pub async fn run<T: AsRef<Path>, U: AsRef<Path>>(
        &self,
        input_filename: T,
        output_filename: U,
        scale: u8,
    ) -> Result<()> {
        let st = Command::new(&self.bin)
            .current_dir(&self.cwd)
            .arg("-i")
            .arg(self.abs_src_dir.join(input_filename))
            .arg("-o")
            .arg(self.abs_dst_dir.join(output_filename))
            .arg("-s")
            .arg(scale.to_string())
            .status()
            .await?;
        if st.success() {
            Ok(())
        } else {
            bail!("child returned {:?}", st);
        }
    }
}
