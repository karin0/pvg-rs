use lru::LruCache;
use std::io::Result;
use std::path::Path;
use std::time::SystemTime;
use tokio::fs;

#[derive(Debug)]
pub struct DiskLru {
    lru: LruCache<String, u64>,
    usage: u64,
}

impl DiskLru {
    pub async fn load(
        size_hint: usize,
        dir: &Path,
        cache_limit: Option<u64>,
    ) -> Result<(Self, Option<u64>)> {
        let mut entires = Vec::with_capacity(size_hint);
        let mut it = fs::read_dir(dir).await?;
        let mut err = false;
        while let Some(file) = it.next_entry().await? {
            let meta = file.metadata().await?;
            if !meta.is_file() {
                error!("not a file: {}", file.path().display());
                err = true;
                continue;
            }

            let size = meta.len();
            if size == 0 {
                error!("empty file: {}", file.path().display());
                err = true;
                continue;
            }

            let file = file.file_name().into_string().unwrap();
            let time = if cache_limit.is_some() {
                meta.accessed()
                    .or_else(|_| meta.modified())
                    .or_else(|_| meta.created())
                    .unwrap_or(SystemTime::UNIX_EPOCH)
            } else {
                SystemTime::UNIX_EPOCH
            };
            entires.push((file, size, time));
        }

        if err {
            error!("Please move away the invalid files, or downloading might fail.");
        }

        let total_size: u64 = entires.iter().map(|(_, size, _)| *size).sum();
        info!(
            "disk: {} files, {:.2} MiB",
            entires.len(),
            total_size as f32 / ((1 << 20) as f32)
        );

        let lru_limit = if let Some(limit) = cache_limit {
            entires.sort_unstable_by_key(|(_, _, time)| time.to_owned());
            if total_size > limit {
                warn!("cache size over limit: {total_size} > {limit}");
                Some(total_size)
            } else {
                Some(limit)
            }
        } else {
            None
        };

        let mut lru = Self::new();
        for (file, size, _) in entires {
            lru.insert(file, size);
        }

        Ok((lru, lru_limit))
    }

    pub fn new() -> Self {
        Self {
            lru: LruCache::unbounded(),
            usage: 0,
        }
    }

    pub fn insert(&mut self, key: String, size: u64) {
        self.lru.push(key, size);
        self.usage += size;
    }

    pub fn evict(&mut self, limit: u64) -> Option<String> {
        if self.usage > limit {
            self.lru.pop_lru().map(|(k, size)| {
                self.usage -= size;
                info!("{}: dropped {:.2} MiB", k, size as f32 / ((1 << 20) as f32));
                k
            })
        } else {
            None
        }
    }

    pub fn promote(&mut self, key: &str) {
        self.lru.promote(key);
    }

    pub fn contains(&self, key: &str) -> bool {
        self.lru.contains(key)
    }

    pub fn filter<F: Fn(&str) -> bool>(&mut self, f: F) -> Vec<String> {
        let a = self
            .lru
            .iter()
            .filter(|(k, _)| !f(k))
            .map(|(k, _)| k.to_owned())
            .collect();
        for k in &a {
            let r = self.lru.pop(k);
            self.usage -= r.unwrap();
        }
        a
    }
}
