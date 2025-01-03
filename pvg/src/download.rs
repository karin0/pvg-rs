use crate::{bug, critical};
use actix_web::web::Bytes;
use anyhow::{bail, Result};
use futures::{stream, Stream};
use std::io;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc::UnboundedSender;
use tokio::time::Instant;

#[derive(Debug, Default)]
struct DownloadGuard(bool);

impl DownloadGuard {
    fn release(&mut self) {
        self.0 = true;
    }
}

impl Drop for DownloadGuard {
    fn drop(&mut self) {
        if !self.0 {
            bug!("unfinished download dropped");
        }
    }
}

#[derive(Debug)]
pub struct DownloadingFile {
    path: PathBuf,
    file: fs::File,
    time: Option<Instant>,
    size: usize,
    guard: DownloadGuard,
}

impl DownloadingFile {
    pub async fn new(path: PathBuf) -> Result<Self> {
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .await?;
        Ok(Self {
            path,
            file,
            time: None,
            size: 0,
            guard: DownloadGuard::default(),
        })
    }

    pub fn start(&mut self) {
        self.time = Some(Instant::now());
    }

    pub async fn write(&mut self, b: &Bytes) -> io::Result<()> {
        self.file.write_all(b).await?;
        self.size += b.len();
        Ok(())
    }

    pub async fn commit(self, path: &Path, size: Option<u64>) -> Result<u64> {
        if let Some(expected) = size {
            if self.size != expected as usize {
                let size = self.size;
                self.rollback().await;
                bail!("expected {} bytes, written {}", expected, size);
            }
        } else {
            debug!("unknown size, written {}", self.size);
        }
        Ok(self.do_commit(path).await?)
    }

    async fn do_commit(mut self, path: &Path) -> io::Result<u64> {
        self.guard.release();
        drop(self.file);
        if let Err(e) = fs::rename(&self.path, path).await {
            critical!("{:?}: COMMIT FAILED: {}", self.path, e);
            Err(e)
        } else {
            if let Some(t) = self.time {
                let t = t.elapsed().as_secs_f32();
                let kib = self.size as f32 / 1024.;
                debug!(
                    "{:?}: committed {:.3} KiB in {:.3} secs ({:.3} KiB/s)",
                    self.path,
                    kib,
                    t,
                    kib / t
                );
            } else {
                debug!("{:?}: committed {} B", self.path, self.size);
            }
            Ok(self.size as u64)
        }
    }

    pub async fn rollback(mut self) {
        self.guard.release();
        drop(self.file);
        if let Err(e) = fs::remove_file(&self.path).await {
            critical!(
                "{:?}: ROLLBACK FAILED ({} bytes): {}",
                self.path,
                self.size,
                e
            );
        } else {
            info!("{:?}: rolled back {} bytes", self.path, self.size);
        }
    }
}

pub struct DownloadingStream {
    pub remote: pixiv::reqwest::Response,
    pub tx: UnboundedSender<Option<Bytes>>,
    pub path: PathBuf,
}

impl DownloadingStream {
    async fn f(mut self) -> Option<(Result<Bytes>, Option<Self>)> {
        match self.remote.chunk().await {
            Ok(Some(b)) => {
                if let Err(e) = self.tx.send(Some(b.clone())) {
                    bug!("{:?}: send error: {}", self.path, e);
                    Some((Err(e.into()), None))
                } else {
                    Some((Ok(b), Some(self)))
                }
            }
            Err(e) => {
                error!("{:?}: remote streaming failed: {}", self.path, e);
                Some((Err(e.into()), None))
            }
            Ok(None) => {
                debug!("{:?}: remote streaming done", self.path);
                if let Err(e) = self.tx.send(None) {
                    bug!("{:?}: done send error: {}", self.path, e);
                }
                None
            }
        }
    }

    async fn fold(this: Option<Self>) -> Option<(Result<Bytes>, Option<Self>)> {
        if let Some(this) = this {
            this.f().await
        } else {
            None
        }
    }

    pub fn stream(self) -> impl Stream<Item = Result<Bytes>> {
        stream::unfold(Some(self), DownloadingStream::fold)
    }
}
