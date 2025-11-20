use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};
use suffix::SuffixTable;

fn binary_search<T, F>(xs: &[T], mut pred: F) -> usize
where
    F: FnMut(&T) -> bool,
{
    let (mut left, mut right) = (0, xs.len());
    while left < right {
        let mid = usize::midpoint(left, right);
        if pred(&xs[mid]) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    left
}

type Key = pixiv::IllustId;
type Pos = u32;
type Item<'a> = (Key, &'a str);

struct SA {
    sa: SuffixTable<'static, 'static>,
    indices: Vec<Pos>,     // len = total length of all strings (in bytes)
    data: Vec<(Key, Pos)>, // len = number of strings
}

static STAT: AtomicU32 = AtomicU32::new(0);

impl SA {
    pub fn from<S: AsRef<str>, I: Iterator<Item = (Key, S)> + Clone>(iter: I) -> SA {
        let size = iter
            .clone()
            .map(|(_, s)| s.as_ref().len() + 1)
            .sum::<usize>()
            .saturating_sub(1);
        let mut full = String::with_capacity(size);
        let mut first = true;
        let data = iter
            .clone()
            .map(|(k, s)| {
                let s = s.as_ref();
                assert!(!s.is_empty());
                if first {
                    first = false;
                } else {
                    full.push('\\');
                }
                let start = full.len() as Pos;
                full.push_str(s);
                (k, start)
            })
            .collect_vec();
        let sa = SuffixTable::new(full);

        let indices = iter
            .enumerate()
            .flat_map(|(idx, (_, s))| {
                std::iter::repeat_n(idx as Pos, s.as_ref().len() + usize::from(idx != 0))
            })
            .collect_vec();

        let mut indices = sa
            .table()
            .iter()
            .map(|&p| indices[p as usize])
            .collect_vec();

        indices.shrink_to_fit();

        SA { sa, indices, data }
    }

    fn indices(&self, query: &str) -> &[Pos] {
        let sa = &self.sa;
        let (text, query) = (sa.text().as_bytes(), query.as_bytes());

        if text.is_empty()
            || query.is_empty()
            || (query < sa.suffix_bytes(0) && !sa.suffix_bytes(0).starts_with(query))
            || query > sa.suffix_bytes(sa.len() - 1)
        {
            return &[];
        }

        let table = sa.table();
        let start = binary_search(table, |&sufi| query <= &text[sufi as usize..]);
        let end = start
            + binary_search(&table[start..], |&sufi| {
                !text[sufi as usize..].starts_with(query)
            });

        if start >= end {
            &[]
        } else {
            &self.indices[start..end]
        }
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
                &full[start as usize..(data[idx + 1].1 - 1) as usize]
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
                if p1 < p2 {
                    (k, &full[p1 as usize..(p2 - 1) as usize])
                } else {
                    (k, &full[p1 as usize..])
                }
            })
    }

    fn items(&self) -> Vec<Item<'_>> {
        self.iter().collect_vec()
    }

    fn single_select<'a>(&'a self, pattern: &str) -> impl Iterator<Item = Key> + 'a {
        self.indices(pattern)
            .iter()
            .sorted_unstable()
            .dedup()
            .map(move |idx| self.data[*idx as usize].0)
    }

    fn heuristic_select<'a>(
        &'a self,
        min_idx: usize,
        min_indices: &'a [Pos],
        filters: &'a [String],
    ) -> impl Iterator<Item = Key> + 'a {
        min_indices
            .iter()
            .sorted_unstable()
            .dedup()
            .filter_map(move |idx| {
                let (s, key) = self.resolve(*idx);
                if filters.iter().enumerate().all(|(p, tag)| {
                    p == min_idx || {
                        STAT.fetch_add(s.len() as u32, Ordering::Relaxed);
                        s.contains(tag)
                    }
                }) {
                    Some(key)
                } else {
                    None
                }
            })
    }

    fn inverted_select<'a>(
        &'a self,
        all_indices: Vec<&'a [Pos]>,
        min_idx: usize,
        min_indices: &'a [Pos],
        ban_filters: &'a [String],
    ) -> Box<dyn Iterator<Item = Key> + 'a> {
        let mut s = min_indices.iter().copied().collect::<HashSet<Pos>>();
        let mut new = HashSet::new();

        for (p, indices) in all_indices.into_iter().enumerate() {
            debug!("candidates: {}", s.len());
            STAT.fetch_add(indices.len() as u32, Ordering::Relaxed);
            if p == min_idx {
                continue;
            }
            for &idx in indices {
                if s.contains(&idx) {
                    new.insert(idx);
                }
            }
            if new.is_empty() {
                return Box::new(std::iter::empty());
            }
            std::mem::swap(&mut s, &mut new);
            new.clear();
        }

        for patt in ban_filters {
            let indices = self.indices(patt);
            STAT.fetch_add(indices.len() as u32, Ordering::Relaxed);
            for idx in indices {
                s.remove(idx);
            }
        }
        Box::new(
            s.into_iter()
                .sorted_unstable()
                .map(move |idx| self.data[idx as usize].0),
        )
    }

    fn select<'a>(
        &'a self,
        filters: &'a [String],
        ban_filters: &'a [String],
    ) -> Box<dyn Iterator<Item = Key> + 'a> {
        match filters.len() {
            0 => {
                // Inverted selection is more suitable for negated filters.
                let ban_idxs = ban_filters
                    .iter()
                    .flat_map(|patt| self.indices(patt))
                    .collect::<HashSet<_>>();
                STAT.fetch_add(ban_idxs.len() as u32, Ordering::Relaxed);
                return Box::new(self.iter().enumerate().filter_map(move |(idx, (key, _))| {
                    if ban_idxs.contains(&(idx as Pos)) {
                        None
                    } else {
                        Some(key)
                    }
                }));
            }
            1 => {
                if ban_filters.is_empty() {
                    return Box::new(self.single_select(&filters[0]));
                }
            }
            _ => {}
        }

        let all: Vec<&[Pos]> = filters
            .iter()
            .map(|patt| self.indices(patt))
            .collect::<Vec<_>>();

        let (p, a) = all
            .iter()
            .copied()
            .enumerate()
            .min_by_key(|(_, poses)| poses.len())
            .unwrap();

        let a_len = a.len();

        if a_len == 0 {
            return Box::new(std::iter::empty());
        }

        let sum_len = if log_enabled!(log::Level::Debug) {
            let lens = all.iter().map(|v| v.len()).collect_vec();
            let sum_len: usize = lens.iter().sum();
            let c = a.len() as f64 / sum_len as f64;
            debug!("indices lengths: {lens:?}, c = {c:.3}");
            if !ban_filters.is_empty() {
                return self.inverted_select(all, p, a, ban_filters);
            }
            sum_len
        } else {
            if !ban_filters.is_empty() {
                return self.inverted_select(all, p, a, ban_filters);
            }
            all.iter().map(|v| v.len()).sum::<usize>()
        };

        // c >= 0.25
        if a.len() * 4 >= sum_len {
            return self.inverted_select(all, p, a, ban_filters);
        }

        Box::new(self.heuristic_select(p, a, filters))
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

    pub fn stat() -> u32 {
        STAT.swap(0, Ordering::Relaxed)
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
