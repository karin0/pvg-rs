use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use suffix::SuffixTable;

type Key = pixiv::IllustId;
type Pos = u32;
type Item<'a> = (Key, &'a str);

struct SA {
    sa: SuffixTable<'static, 'static>,
    indices: Vec<Pos>,     // len = total length of all strings (in bytes)
    data: Vec<(Key, Pos)>, // len = number of strings
}

impl SA {
    pub fn from<S: AsRef<str>, I: Iterator<Item = (Key, S)> + Clone>(iter: I) -> SA {
        let mut indices = iter
            .clone()
            .enumerate()
            .flat_map(|(idx, (_, s))| std::iter::repeat_n(idx as Pos, s.as_ref().len()))
            .collect_vec();

        let size = iter.clone().map(|(_, s)| s.as_ref().len()).sum();
        let mut full = String::with_capacity(size);
        let data = iter
            .map(|(k, s)| {
                let start = full.len() as Pos;
                full.push_str(s.as_ref());
                (k, start)
            })
            .collect_vec();
        let sa = SuffixTable::new(full);

        indices.shrink_to_fit();
        SA { sa, indices, data }
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn size(&self) -> usize {
        self.indices.len()
    }

    fn memory(&self) -> usize {
        use std::mem::size_of;
        let n: usize = self.size();
        // SuffixTable: The space usage is roughly `6` bytes per character.
        n * (6 + size_of::<Pos>() + size_of::<u8>())
            + self.data.len() * (size_of::<Key>() + size_of::<Pos>())
            + size_of::<Self>()
    }

    fn resolve(&self, idx: u32) -> (&str, Key) {
        let idx = idx as usize;
        let data = &self.data;
        let (key, start) = data[idx];
        let full = self.sa.text();
        (
            if idx + 1 < self.len() {
                &full[start as usize..data[idx + 1].1 as usize]
            } else {
                &full[start as usize..]
            },
            key,
        )
    }

    fn iter(&self) -> impl Iterator<Item = Item<'_>> {
        let full = self.sa.text();
        self.data
            .iter()
            .circular_tuple_windows()
            .map(|(&(k, p1), &(_, p2))| {
                if p1 <= p2 {
                    (k, &full[p1 as usize..p2 as usize])
                } else {
                    (k, &full[p1 as usize..])
                }
            })
    }

    fn items(&self) -> Vec<Item<'_>> {
        self.iter().collect_vec()
    }

    fn single_select<'a>(
        &'a self,
        pattern: &str,
        ban_filters: &'a [String],
    ) -> impl Iterator<Item = Key> + 'a {
        self.sa
            .positions(pattern)
            .iter()
            .sorted_unstable()
            .map(|i| self.indices[*i as usize])
            .dedup()
            .filter_map(move |idx| {
                let (s, key) = self.resolve(idx);
                if ban_filters.iter().all(|tag| !s.contains(tag)) {
                    Some(key)
                } else {
                    None
                }
            })
    }

    fn _select_best<'a>(
        &'a self,
        filters: &'a [String],
        ban_filters: &'a [String],
    ) -> impl Iterator<Item = Key> + 'a {
        let (i, a) = filters
            .iter()
            .map(|patt| self.sa.positions(patt))
            .enumerate()
            .min_by_key(|(_, poses)| poses.len())
            .unwrap();
        a.iter()
            .sorted_unstable()
            .map(|p| self.indices[*p as usize])
            .dedup()
            .filter_map(move |idx| {
                let (s, key) = self.resolve(idx);
                if ban_filters.iter().all(|tag| !s.contains(tag))
                    && filters
                        .iter()
                        .enumerate()
                        .all(|(p, tag)| p == i || s.contains(tag))
                {
                    Some(key)
                } else {
                    None
                }
            })
    }

    fn select<'a>(
        &'a self,
        filters: &'a [String],
        ban_filters: &'a [String],
    ) -> Box<dyn Iterator<Item = Key> + 'a> {
        match filters.len() {
            0 => Box::new(self.iter().filter_map(move |(key, s)| {
                if ban_filters.iter().all(|tag| !s.contains(tag)) {
                    Some(key)
                } else {
                    None
                }
            })),
            1 => Box::new(self.single_select(&filters[0], ban_filters)),
            _ => Box::new(self._select_best(filters, ban_filters)),
        }
    }
}

impl Default for SA {
    fn default() -> Self {
        Self::from::<&str, _>(std::iter::empty())
    }
}

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
                let idx = out.len() as Pos;
                out.push(item);
                ent.insert(idx);
            }
        }
    }

    fn extend<'a, I: IntoIterator<Item = Item<'a>>>(&mut self, out: &mut Vec<Item<'a>>, iter: I) {
        for (k, s) in iter {
            self.push(out, (k, s));
        }
    }
}

fn extend<'a, I: IntoIterator<Item = Item<'a>>>(out: &mut Vec<Item<'a>>, iter: I) {
    for (k, s) in iter {
        out.push((k, s));
    }
}

#[derive(Default)]
pub struct Index {
    major: SA,
    minor: Option<SA>,
    dedup: bool,
}

impl Index {
    pub fn new<S: AsRef<str>, I: Iterator<Item = (Key, S)> + Clone>(data: I) -> Self {
        Self {
            major: SA::from(data),
            minor: None,
            dedup: false,
        }
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

    fn _select<I: Iterator<Item = Key>>(&self, iter: I) -> Vec<Key> {
        if self.dedup {
            let mut seen = HashSet::new();
            iter.filter(|k| seen.insert(*k)).collect_vec()
        } else {
            iter.collect_vec()
        }
    }

    pub fn select(&self, filters: &[String], ban_filters: &[String]) -> Vec<Key> {
        if let Some(minor) = &self.minor {
            self._select(
                self.major
                    .select(filters, ban_filters)
                    .chain(minor.select(filters, ban_filters)),
            )
        } else {
            self._select(self.major.select(filters, ban_filters))
        }
    }

    // `dup` should be set if provided `data` may contain duplicates of existing items.
    pub fn insert(&mut self, data: Vec<(Key, String)>, dup: bool) {
        // The entire minor SA is "infected" if `dup` is set.
        self.dedup |= dup;
        if let Some(minor) = self.minor.take() {
            let len = data.len();
            let data = data.iter().map(|(k, s)| (*k, s.as_str()));
            if (minor.len() + data.len()) * 16 > self.major.len() {
                warn!(
                    "Merging SA indices: major={}, minor={}, new={}",
                    self.major.len(),
                    minor.len(),
                    data.len()
                );
                let mut combined = self.major.items();
                combined.reserve(minor.len() + len);
                if self.dedup {
                    let mut dedup = Dedup::default();
                    dedup.extend(&mut combined, minor.iter());
                    if dup {
                        dedup.extend(&mut combined, data);
                    } else {
                        extend(&mut combined, data);
                    }
                    self.dedup = false;
                } else {
                    combined.extend(minor.iter());
                    extend(&mut combined, data);
                }
                self.major = SA::from(combined.into_iter());
                self.minor = None;
            } else {
                // let mut combined = minor.data;
                let mut combined = minor.items();
                combined.reserve(len);
                if dup {
                    let mut dedup = Dedup::default();
                    dedup.extend(&mut combined, data);
                } else {
                    extend(&mut combined, data);
                }
                self.minor = Some(SA::from(combined.into_iter()));
            }
        } else {
            self.minor = Some(SA::from(data.into_iter()));
        }
    }
}
