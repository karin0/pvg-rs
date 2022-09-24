use lru::LruCache;

#[derive(Debug)]
pub struct DiskLru {
    lru: LruCache<String, u64>,
    usage: u64,
}

impl DiskLru {
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
}
