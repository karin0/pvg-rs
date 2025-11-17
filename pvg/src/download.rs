use crate::{bug, critical};
use actix_web::web::Bytes;
use anyhow::{Result, bail};
use futures::{Stream, stream};
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
                bail!("expected {expected} bytes, written {size}");
            }
        } else {
            debug!("unknown size, written {}", self.size);
        }

        #[cfg(not(feature = "rename2"))]
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .await
        {
            Ok(f) => drop(f),
            Err(e) => {
                self.rollback().await;
                bail!("failed to open target {:?}: {}", path, e);
            }
        }

        Ok(self.do_commit(path).await?)
    }

    async fn do_commit(mut self, path: &Path) -> io::Result<u64> {
        self.guard.release();
        drop(self.file);

        #[cfg(not(feature = "rename2"))]
        let res = fs::rename(&self.path, path).await;

        #[cfg(feature = "rename2")]
        let res = {
            use nix::fcntl::{AT_FDCWD, RenameFlags, renameat2};

            let res = renameat2(
                AT_FDCWD,
                &self.path,
                AT_FDCWD,
                path,
                RenameFlags::RENAME_NOREPLACE,
            );
            debug!(
                "{} -> {}: renameat2 {:?}",
                self.path.display(),
                path.display(),
                res
            );

            res.map_err(|e| io::Error::from_raw_os_error(e as i32))
        };

        if let Err(e) = res {
            critical!("{:?}: COMMIT FAILED: {}", self.path, e);
            Self::do_rollback(&self.path, self.size).await;
            Err(e)
        } else {
            if let Some(t) = self.time {
                let t = t.elapsed().as_secs_f32();
                let kib = self.size as f32 / 1024.;
                debug!(
                    "{}: committed {:.3} KiB in {:.3} secs ({:.3} KiB/s)",
                    self.path.display(),
                    kib,
                    t,
                    kib / t
                );
            } else {
                debug!("{}: committed {} B", self.path.display(), self.size);
            }
            Ok(self.size as u64)
        }
    }

    pub async fn rollback(mut self) {
        self.guard.release();
        drop(self.file);
        Self::do_rollback(&self.path, self.size).await;
    }

    async fn do_rollback(path: &Path, size: usize) {
        if let Err(e) = fs::remove_file(&path).await {
            critical!("{:?}: ROLLBACK FAILED ({} bytes): {}", path, size, e);
        } else {
            warn!("{}: rolled back {size} bytes", path.display());
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
                error!("{}: remote streaming failed: {}", self.path.display(), e);
                Some((Err(e.into()), None))
            }
            Ok(None) => {
                debug!("{}: remote streaming done", self.path.display());
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
