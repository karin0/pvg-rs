use fixedbitset::FixedBitSet;
use integer_sqrt::IntegerSquareRoot;
use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::mem::size_of;
use std::sync::atomic::{AtomicU32, Ordering};
use suffix::SuffixTable;

// https://github.com/BurntSushi/suffix/blob/5ba4f72941872b697ff3c216f8315ff6de4bf5d7/src/table.rs
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
type Idx = u32;
type Item<'a> = (Key, &'a str);

type Range = (Pos, Pos);

trait RangeExt {
    fn len(&self) -> usize;
}

impl RangeExt for Range {
    fn len(&self) -> usize {
        (self.1 - self.0) as usize
    }
}

struct SA {
    sa: SuffixTable<'static, 'static>,
    indices: Vec<Idx>, // len = N, total length of all strings (in bytes)
    data: Vec<Key>,    // len = V, number of strings
    starts: Vec<Pos>,  // len = V
    #[cfg(feature = "sa-inverted")]
    occurrences: Vec<Vec<Pos>>, // len = V, but total size of inner vecs = N
    blocks: Vec<FixedBitSet>, // len ~ sqrt(N)
    block_size: Pos,
}

#[cfg(feature = "sa-bench")]
static FLAGS: AtomicU32 = AtomicU32::new(0);

static STAT: AtomicU32 = AtomicU32::new(0);

impl SA {
    fn memory(&self) -> usize {
        let n = self.size();

        let r = n * (size_of::<Pos>() + size_of::<Idx>() + size_of::<u8>())
            + self.data.len()
                * (size_of::<Key>()
                    + size_of::<Pos>()
                    + self.blocks.len() / size_of::<fixedbitset::Block>())
            + size_of::<Self>();

        #[cfg(feature = "sa-bench")]
        return r + n;
        #[cfg(not(feature = "sa-bench"))]
        r
    }

    fn stat(&self, dt: std::time::Duration) {
        let n = self.size();
        let v = self.len();
        let sa = (n * size_of::<Pos>()) >> 10;
        let blks = (self.blocks.len() * v / size_of::<fixedbitset::Block>()) >> 10;
        let tot = self.memory() >> 10;
        let b = self.block_size;
        let d = if v > 0 { n / v } else { 0 };
        debug!(
            "SA: sa=indices={sa}, blocks={blks}, total={tot} KiB (n={n}={v}*{d}, b={b}) in {dt:?}"
        );
    }

    pub fn from<S: AsRef<str>, I: Iterator<Item = (Key, S)> + Clone>(iter: I) -> SA {
        let t0 = std::time::Instant::now();
        let size = iter
            .clone()
            .map(|(_, s)| s.as_ref().len() + 1)
            .sum::<usize>()
            .saturating_sub(1);
        let mut full = String::with_capacity(size);
        let mut first = true;
        let (mut data, mut starts): (Vec<Key>, Vec<Pos>) = iter
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
            .unzip();
        data.shrink_to_fit();
        starts.shrink_to_fit();

        let t1 = std::time::Instant::now();
        let sa = SuffixTable::new(full);
        debug!("SA constructed in {:?}", t1.elapsed());

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

        #[cfg(feature = "sa-inverted")]
        let occurrences = {
            let mut occurrences: Vec<Vec<Pos>> = vec![Vec::new(); data.len()];
            for (i, &idx) in indices.iter().enumerate() {
                occurrences[idx as usize].push(i as Pos);
            }
            for occ in &mut occurrences {
                occ.shrink_to_fit();
            }
            occurrences.shrink_to_fit();
            occurrences
        };

        let n = indices.len() as Idx;
        let (mut blocks, block_size) = if n > 0 {
            let block_size = n.integer_sqrt().clamp(1, 10000);
            let v = data.len();
            let blocks = indices
                .chunks(block_size as usize)
                .map(|chunk| {
                    let mut set = FixedBitSet::with_capacity(v);
                    for &idx in chunk {
                        set.insert(idx as usize);
                    }
                    set
                })
                .collect_vec();

            (blocks, block_size)
        } else {
            (Vec::new(), 0)
        };
        blocks.shrink_to_fit();

        let r = SA {
            sa,
            indices,
            data,
            starts,
            #[cfg(feature = "sa-inverted")]
            occurrences,
            blocks,
            block_size,
        };

        if n > 0 && log_enabled!(log::Level::Debug) {
            r.stat(t0.elapsed());
        }

        r
    }

    // https://github.com/BurntSushi/suffix/blob/5ba4f72941872b697ff3c216f8315ff6de4bf5d7/src/table.rs
    fn indices_range(&self, query: &str) -> Range {
        let sa = &self.sa;
        let (text, query) = (sa.text().as_bytes(), query.as_bytes());

        if text.is_empty()
            || query.is_empty()
            || (query < sa.suffix_bytes(0) && !sa.suffix_bytes(0).starts_with(query))
            || query > sa.suffix_bytes(sa.len() - 1)
        {
            return (0, 0);
        }

        let table = sa.table();
        let start = binary_search(table, |&sufi| query <= &text[sufi as usize..]);
        let end = start
            + binary_search(&table[start..], |&sufi| {
                !text[sufi as usize..].starts_with(query)
            });

        if start >= end {
            (0, 0)
        } else {
            (start as Pos, end as Pos)
        }
    }

    fn indices_from_range(&self, range: Range) -> &[Idx] {
        &self.indices[range.0 as usize..range.1 as usize]
    }

    fn indices(&self, pattern: &str) -> &[Idx] {
        self.indices_from_range(self.indices_range(pattern))
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn size(&self) -> usize {
        self.indices.len()
    }

    fn resolve(&self, idx: u32) -> (&str, Key) {
        let idx = idx as usize;
        let starts = &self.starts;
        let full = self.sa.text();
        (
            if idx + 1 < self.len() {
                &full[starts[idx] as usize..(starts[idx + 1] - 1) as usize]
            } else {
                &full[starts[idx] as usize..]
            },
            self.data[idx],
        )
    }

    fn iter(&self) -> impl Iterator<Item = Item<'_>> {
        let full = self.sa.text();
        let strs = self
            .starts
            .iter()
            .circular_tuple_windows()
            .map(|(&p1, &p2)| {
                if p1 < p2 {
                    &full[p1 as usize..(p2 - 1) as usize]
                } else {
                    &full[p1 as usize..]
                }
            });
        self.data.iter().copied().zip(strs)
    }

    fn items(&self) -> Vec<Item<'_>> {
        self.iter().collect_vec()
    }

    fn single_select<'a>(&'a self, pattern: &str) -> impl Iterator<Item = Key> + 'a {
        self.indices(pattern)
            .iter()
            .sorted_unstable()
            .dedup()
            .map(move |idx| self.data[*idx as usize])
    }

    fn heuristic_select<'a>(
        &'a self,
        min_idx: Idx,
        min_indices: &'a [Pos],
        filters: &'a [String],
        ban_filters: &'a [String],
    ) -> impl Iterator<Item = Key> + 'a {
        min_indices
            .iter()
            .sorted_unstable()
            .dedup()
            .filter_map(move |idx| {
                let (s, key) = self.resolve(*idx);
                let l = s.len();
                if filters.iter().enumerate().all(|(p, tag)| {
                    p as Idx == min_idx || {
                        STAT.fetch_add(l as u32, Ordering::Relaxed);
                        s.contains(tag)
                    }
                } && !ban_filters.iter().any(|tag| {
                    STAT.fetch_add(l as u32, Ordering::Relaxed);
                    s.contains(tag)
                })) {
                    Some(key)
                } else {
                    None
                }
            })
    }

    #[cfg(feature = "sa-inverted")]
    fn occured_in_range(&self, idx: Idx, range: Range) -> bool {
        let occurences = &self.occurrences[idx as usize];
        if let Err(bound) = occurences.binary_search(&range.0) {
            occurences.get(bound).is_some_and(|&p| p < range.1)
        } else {
            true
        }
    }

    #[cfg(feature = "sa-inverted")]
    fn binary_select<'a>(
        &'a self,
        all_ranges: Vec<Range>,
        min_idx: Idx,
        min_indices: &'a [Pos],
        ban_filters: &'a [String],
    ) -> impl Iterator<Item = Key> + 'a {
        let ban_ranges = self.query_ranges(ban_filters).collect_vec();
        min_indices
            .iter()
            .sorted_unstable()
            .dedup()
            .copied()
            .filter_map(move |idx| {
                if all_ranges
                    .iter()
                    .enumerate()
                    .all(|(p, &range)| p as Idx == min_idx || self.occured_in_range(idx, range))
                    && !ban_ranges
                        .iter()
                        .any(|&range| self.occured_in_range(idx, range))
                {
                    Some(self.data[idx as usize])
                } else {
                    None
                }
            })
    }

    #[cfg(feature = "sa-bench")]
    fn inverted_select<'a>(
        &'a self,
        all_ranges: Vec<Range>,
        min_idx: Idx,
        min_indices: &'a [Pos],
        ban_filters: &'a [String],
    ) -> Box<dyn Iterator<Item = Key> + 'a> {
        let mut s = min_indices.iter().copied().collect::<HashSet<Pos>>();
        let mut new = HashSet::new();

        for (p, range) in all_ranges.into_iter().enumerate() {
            debug!("candidates: {}", s.len());
            STAT.fetch_add(range.len() as u32, Ordering::Relaxed);
            if p as Idx == min_idx {
                continue;
            }
            let indices = &self.indices[range.0 as usize..range.1 as usize];
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

        for range in ban_filters {
            let indices = self.indices(range);
            STAT.fetch_add(indices.len() as u32, Ordering::Relaxed);
            for idx in indices {
                s.remove(idx);
            }
        }
        Box::new(
            s.into_iter()
                .sorted_unstable()
                .map(move |idx| self.data[idx as usize]),
        )
    }

    #[cfg(feature = "sa-bench")]
    fn inverted_ban_select<'a>(
        &'a self,
        ban_filters: &'a [String],
    ) -> impl Iterator<Item = Key> + 'a {
        // Inverted selection is more suitable for negated filters.
        let ban_idxs = ban_filters
            .iter()
            .flat_map(|patt| self.indices(patt))
            .collect::<HashSet<_>>();
        STAT.fetch_add(ban_idxs.len() as u32, Ordering::Relaxed);
        self.iter().enumerate().filter_map(move |(idx, (key, _))| {
            if ban_idxs.contains(&(idx as Pos)) {
                None
            } else {
                Some(key)
            }
        })
    }

    fn join_blocks(&self, l: Pos, r: Pos) -> FixedBitSet {
        let b = self.block_size;
        let start = l.div_ceil(b);
        let end = r / b;
        let mut res = FixedBitSet::with_capacity(self.len());

        let bound = r.min(start * b);
        if l < bound {
            for &idx in &self.indices[l as usize..bound as usize] {
                res.insert(idx as usize);
            }
        }
        if r <= bound {
            return res;
        }
        let bound = l.max(end * b);
        if bound < r {
            for &idx in &self.indices[bound as usize..r as usize] {
                res.insert(idx as usize);
            }
        }
        if start < end {
            let blocks = end - start;
            STAT.fetch_add(blocks, Ordering::Relaxed);
            debug!("joining {blocks} blocks");
            for b in &self.blocks[start as usize..end as usize] {
                res.union_with(b);
            }
        }

        res
    }

    fn block_select(
        &self,
        ranges: impl Iterator<Item = Range>,
        ban_filters: &[String],
    ) -> impl Iterator<Item = Key> {
        let v = self.len();
        let mut res = FixedBitSet::with_capacity(v);
        res.insert_range(0..v);
        for (l, r) in ranges {
            res.intersect_with(&self.join_blocks(l, r));
        }
        for f in ban_filters {
            let (l, r) = self.indices_range(f);
            res.difference_with(&self.join_blocks(l, r));
        }
        res.ones()
            .map(|idx| self.data[idx])
            .collect::<Vec<_>>()
            .into_iter()
    }

    fn block_ban_select(
        &self,
        ban_ranges: impl Iterator<Item = Range>,
    ) -> impl Iterator<Item = Key> {
        let v = self.len();
        let mut set = FixedBitSet::with_capacity(v);
        for (l, r) in ban_ranges {
            set.union_with(&self.join_blocks(l, r));
        }
        self.iter().enumerate().filter_map(
            move |(idx, (key, _))| {
                if set.contains(idx) { None } else { Some(key) }
            },
        )
    }

    #[cfg(feature = "sa-bench")]
    fn brute_select(
        &self,
        ranges: impl Iterator<Item = Range>,
        ban_ranges: impl Iterator<Item = Range>,
    ) -> impl Iterator<Item = Key> {
        use std::collections::BTreeSet;
        let mut s = BTreeSet::new();
        s.extend(0..self.len() as Pos);
        for range in ranges {
            let indices = self.indices_from_range(range);
            let t = indices.iter().copied().collect::<BTreeSet<_>>();
            STAT.fetch_add(t.len() as u32, Ordering::Relaxed);
            s = s.intersection(&t).copied().collect();
        }
        for range in ban_ranges {
            let indices = self.indices_from_range(range);
            let t: BTreeSet<u32> = indices.iter().copied().collect::<BTreeSet<_>>();
            STAT.fetch_add(t.len() as u32, Ordering::Relaxed);
            s = s.difference(&t).copied().collect();
        }
        s.into_iter().map(|idx| self.data[idx as usize])
    }

    fn query_ranges(&self, patterns: &[String]) -> impl Iterator<Item = Range> {
        patterns.iter().map(|patt| self.indices_range(patt))
    }

    fn select<'a>(
        &'a self,
        filters: &'a [String],
        ban_filters: &'a [String],
    ) -> Box<dyn Iterator<Item = Key> + 'a> {
        #[cfg(feature = "sa-bench")]
        let flags = FLAGS.load(Ordering::Relaxed) & 0xf;

        #[cfg(feature = "sa-bench")]
        if flags == 0x9 {
            debug!("forced brute select");
            return Box::new(
                self.brute_select(self.query_ranges(filters), self.query_ranges(ban_filters)),
            );
        }

        if self.block_size == 0 {
            return Box::new(std::iter::empty());
        }

        match filters.len() {
            0 => {
                #[cfg(feature = "sa-bench")]
                if flags == 0x5 {
                    debug!("forced inverted ban select");
                    return Box::new(self.inverted_ban_select(ban_filters));
                }
                return Box::new(self.block_ban_select(self.query_ranges(ban_filters)));
            }
            1 => {
                #[cfg(feature = "sa-bench")]
                let cond = flags == 0 && ban_filters.is_empty();
                #[cfg(not(feature = "sa-bench"))]
                let cond = ban_filters.is_empty();
                if cond {
                    return Box::new(self.single_select(&filters[0]));
                }
            }
            _ => {}
        }

        let all = self.query_ranges(filters);

        #[cfg(feature = "sa-bench")]
        if flags == 0x7 {
            debug!("forced block select");
            return Box::new(self.block_select(all, ban_filters));
        }

        let all = all.collect_vec();
        let (min_idx, &min) = all
            .iter()
            .enumerate()
            .min_by_key(|(_, poses)| poses.len())
            .unwrap();

        let min_idx = min_idx as Idx;
        let min_len = min.len();

        if min_len == 0 {
            return Box::new(std::iter::empty());
        }

        let min_indices = self.indices_from_range(min);

        #[cfg(feature = "sa-bench")]
        if flags == 0x1 {
            debug!("forced heuristic selection");
            return Box::new(self.heuristic_select(min_idx, min_indices, filters, ban_filters));
        }

        if log_enabled!(log::Level::Debug) {
            let lens = all.iter().map(RangeExt::len).collect_vec();
            let sum_len: usize = lens.iter().sum();
            let c = min_len as f64 / sum_len as f64;
            debug!("indices lengths: {lens:?}, c = {c:.6}");
        }

        #[cfg(feature = "sa-bench")]
        if flags == 0x5 {
            debug!("forced inverted selection");
            return self.inverted_select(all, min_idx, min_indices, ban_filters);
        }

        #[cfg(all(feature = "sa-bench", feature = "sa-inverted"))]
        if flags == 0x3 {
            debug!("forced binary selection");
            return Box::new(self.binary_select(all, min_idx, min_indices, ban_filters));
        }

        if min_len <= 150 {
            #[cfg(feature = "sa-inverted")]
            return {
                let pattern_len: usize = filters
                    .iter()
                    .chain(ban_filters.iter())
                    .map(String::len)
                    .sum();
                if pattern_len >= 80 {
                    Box::new(self.binary_select(all, min_idx, min_indices, ban_filters))
                } else {
                    Box::new(self.heuristic_select(min_idx, min_indices, filters, ban_filters))
                }
            };
            #[cfg(not(feature = "sa-inverted"))]
            return Box::new(self.heuristic_select(min_idx, min_indices, filters, ban_filters));
        }

        Box::new(self.block_select(all.into_iter(), ban_filters))
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

    #[cfg(feature = "sa-bench")]
    pub fn set_flags(flags: u32) {
        FLAGS.store(flags, Ordering::Relaxed);
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
