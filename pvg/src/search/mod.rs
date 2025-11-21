mod indices;
mod segment;
mod types;

#[cfg(feature = "fm-index")]
mod fm_index;

use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;

pub use segment::Query;
use segment::Segment;
use types::{Item, Key, Pos, STATS};

#[cfg(feature = "sa-bench")]
use segment::FLAGS;

#[derive(Default)]
struct Dedup(HashMap<Key, Pos>);

impl Dedup {
    fn push<'a>(&mut self, out: &mut Vec<Item<'a>>, item: Item<'a>) {
        use std::collections::hash_map::Entry;
        let key = item.0;
        match self.0.entry(key) {
            Entry::Occupied(ent) => {
                let idx = *ent.get() as usize;
                out[idx] = item;
            }
            Entry::Vacant(ent) => {
                ent.insert(out.len() as Pos);
                out.push(item);
            }
        }
    }

    fn extend<'a, I: IntoIterator<Item = Item<'a>>>(&mut self, out: &mut Vec<Item<'a>>, iter: I) {
        for (k, s) in iter {
            self.push(out, (k, s));
        }
    }
}

#[derive(Default)]
pub struct Index {
    major: Segment,
    minor: Option<Segment>,
    dedup: bool,
}

impl Index {
    pub fn new<S: AsRef<str>, I: Iterator<Item = (Key, S)>>(data: I) -> Self {
        Self {
            major: Segment::from_iter(data),
            minor: None,
            dedup: false,
        }
    }

    #[cfg(feature = "sa-bench")]
    pub fn set_flags(flags: u32) {
        FLAGS.store(flags, Ordering::Relaxed);
    }

    pub fn stats() -> u32 {
        STATS.swap(0, Ordering::Relaxed)
    }

    pub fn size(&self) -> usize {
        let mut sz = self.major.size();
        if let Some(minor) = &self.minor {
            sz += minor.size();
        }
        sz
    }

    pub fn memory(&self) -> usize {
        let mut sz = self.major.memory();
        if let Some(minor) = &self.minor {
            sz += minor.memory();
        }
        sz
    }

    fn do_select<I: Iterator<Item = Key>>(&self, iter: I) -> Vec<Key> {
        if self.dedup {
            let mut seen = HashSet::new();
            iter.filter(|k| seen.insert(*k)).collect_vec()
        } else {
            iter.collect_vec()
        }
    }

    pub fn select(&self, query: Query) -> Vec<Key> {
        if let Some(minor) = &self.minor {
            self.do_select(self.major.select(query).chain(minor.select(query)))
        } else {
            self.do_select(self.major.select(query))
        }
    }

    // `dup` should be set if provided `data` may contain duplicates of existing items.
    pub fn insert(&mut self, data: Vec<(Key, String)>, dup: bool) {
        // The entire minor SA is "infected" if `dup` is set.
        self.dedup |= dup;
        if let Some(minor) = self.minor.take() {
            let len = data.len();
            let data = data.iter().map(|(k, s)| (*k, s.as_str()));
            let minor_len = minor.len();
            if (minor_len + data.len()) * 16 > self.major.len() {
                warn!(
                    "Compacting SA indices: major={}, minor={minor_len}, new={}",
                    self.major.len(),
                    data.len()
                );

                // Drop the old indices before reconstruction.
                let minor = minor.into_inner();
                self.minor = None;

                self.major = if self.dedup {
                    let major = std::mem::take(&mut self.major).into_inner();
                    let mut combined = major.iter().collect_vec();
                    combined.reserve(minor_len + len);

                    let mut dedup = Dedup::default();
                    dedup.extend(&mut combined, minor.iter());
                    if dup {
                        dedup.extend(&mut combined, data);
                    } else {
                        combined.extend(data);
                    }

                    self.dedup = false;
                    Segment::from_iter(combined.into_iter())
                } else {
                    std::mem::take(&mut self.major).extend(minor.iter().chain(data))
                }
            } else {
                self.minor = Some(if dup {
                    let minor = minor.into_inner();
                    let mut combined = minor.iter().collect_vec();
                    combined.reserve(len);

                    let mut dedup = Dedup::default();
                    dedup.extend(&mut combined, data);

                    Segment::from_iter(combined.into_iter())
                } else {
                    minor.extend(data)
                });
            }
        } else {
            self.minor = Some(Segment::from_iter(data.into_iter()));
        }
    }
}
