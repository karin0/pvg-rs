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

#[derive(Debug)]
pub struct DownloadingFile {
    path: PathBuf,
    file: Option<fs::File>,
    time: Option<Instant>,
    size: usize,
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
            file: Some(file),
            time: None,
            size: 0,
        })
    }

    pub fn start(&mut self) {
        self.time = Some(Instant::now());
    }

    pub async fn write(&mut self, b: &Bytes) -> io::Result<()> {
        let Some(ref mut file) = self.file else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "write to closed DownloadingFile",
            ));
        };
        file.write_all(b).await?;
        self.size += b.len();
        Ok(())
    }

    pub async fn commit(mut self, path: &Path, size: Option<u64>) -> Result<u64> {
        drop(self.file.take());
        if let Some(expected) = size {
            if self.size != expected as usize {
                let size = self.size;
                self.rollback().await;
                bail!("expected {expected} bytes, written {size}");
            }
        } else {
            debug!("unknown size, written {}", self.size);
        }

        #[cfg(feature = "rename2")]
        {
            use log::Level::{Debug, Error};
            use nix::fcntl::{AT_FDCWD, RenameFlags, renameat2};

            let res = renameat2(
                AT_FDCWD,
                &self.path,
                AT_FDCWD,
                path,
                RenameFlags::RENAME_NOREPLACE,
            );
            log!(
                if res.is_ok() { Debug } else { Error },
                "{} -> {}: renameat2: {:?}",
                self.path.display(),
                path.display(),
                res
            );

            // Falling back to fs::rename if renameat2 fails.
            if res.is_ok() {
                return Ok(self.finalize());
            }
        };

        // Create a new file to ensure that the target does not exist.
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .await
        {
            Ok(f) => drop(f),
            Err(e) => {
                self.rollback().await;
                bail!("failed to open target {}: {e:?}", path.display());
            }
        }

        let res = fs::rename(&self.path, path).await;

        if let Err(e) = res {
            critical!("{:?}: COMMIT FAILED: {}", self.path, e);
            self.rollback().await;
            Err(e.into())
        } else {
            Ok(self.finalize())
        }
    }

    fn finalize(self) -> u64 {
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
        self.size as u64
    }

    pub async fn rollback(mut self) {
        drop(self.file.take());
        Self::rollback_finalize(&self.path, self.size, fs::remove_file(&self.path).await);
    }

    fn rollback_finalize(path: &Path, size: usize, res: io::Result<()>) {
        if let Err(e) = res {
            critical!("{:?}: ROLLBACK FAILED ({} bytes): {}", path, size, e);
        } else {
            warn!("{}: rolled back {size} bytes", path.display());
        }
    }
}

impl Drop for DownloadingFile {
    fn drop(&mut self) {
        let Some(file) = self.file.take() else {
            return;
        };
        bug!("{}: unfinalized DownloadingFile", self.path.display());
        match file.try_into_std() {
            Ok(file) => drop(file),
            Err(e) => {
                error!("{}: try_into_std failed on {:?}", self.path.display(), e);
            }
        }
        Self::rollback_finalize(&self.path, self.size, std::fs::remove_file(&self.path));
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
