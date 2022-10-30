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
