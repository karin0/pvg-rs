use fixedbitset::FixedBitSet;
use integer_sqrt::IntegerSquareRoot;
use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::mem::size_of;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};
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
    #[inline]
    fn len(&self) -> usize {
        (self.1 - self.0) as usize
    }
}

struct Indices {
    inner: Vec<Idx>, // len = N, total length of all strings (in bytes)
    #[cfg(feature = "sa-inverted")]
    occurrences: Vec<Vec<Pos>>, // len = V, but total size of inner vecs = N
    blocks: Vec<FixedBitSet>, // len ~ sqrt(N)
    block_size: Pos,
}

impl Indices {
    fn new<S: AsRef<str>, I: Iterator<Item = S>>(iter: I, v: Idx, sa: &[Pos]) -> Indices {
        let raw_indices = iter
            .enumerate()
            .flat_map(|(idx, s)| {
                std::iter::repeat_n(idx as Pos, s.as_ref().len() + usize::from(idx != 0))
            })
            .collect_vec();

        let mut indices = sa.iter().map(|&p| raw_indices[p as usize]).collect_vec();
        drop(raw_indices);
        indices.shrink_to_fit();

        #[cfg(feature = "sa-inverted")]
        let occurrences = {
            let mut occurrences: Vec<Vec<Pos>> = vec![Vec::new(); v as usize];
            for (i, &idx) in indices.iter().enumerate() {
                occurrences[idx as usize].push(i as Pos);
            }
            for occ in &mut occurrences {
                occ.shrink_to_fit();
            }
            occurrences.shrink_to_fit();
            occurrences
        };

        let n = indices.len() as Pos;
        let (mut blocks, block_size) = if n > 0 {
            let block_size = n.integer_sqrt().clamp(1, 10000);
            let blocks = indices
                .chunks(block_size as usize)
                .map(|chunk| {
                    let mut set = FixedBitSet::with_capacity(v as usize);
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

        Indices {
            inner: indices,
            #[cfg(feature = "sa-inverted")]
            occurrences,
            blocks,
            block_size,
        }
    }

    #[inline]
    fn join_blocks(&self, range: Range, v: Idx) -> FixedBitSet {
        let (l, r) = range;
        let b = self.block_size;
        let start = l.div_ceil(b);
        let end = r / b;

        let bound = r.min(start * b);
        if r <= bound {
            // Same block, r == bound.
            let mut res = FixedBitSet::with_capacity(v as usize);
            for &idx in &self.inner[l as usize..r as usize] {
                // Safety: `res.len()` == `v`, and all indices are < `v`.
                unsafe {
                    res.insert_unchecked(idx as usize);
                }
            }
            return res;
        }

        let mut res;
        if start < end {
            let blocks = end - start;
            STATS.fetch_add(blocks, Ordering::Relaxed);
            debug!("joining {blocks} blocks");

            // Safety: `start < end` ensures at least one block exists.
            let (first, rest) = unsafe {
                self.blocks[start as usize..end as usize]
                    .split_first()
                    .unwrap_unchecked()
            };

            res = first.clone();
            for b in rest {
                res.union_with(b);
            }
        } else {
            res = FixedBitSet::with_capacity(v as usize);
        }

        for &idx in &self.inner[l as usize..bound as usize] {
            unsafe {
                res.insert_unchecked(idx as usize);
            }
        }

        let bound = l.max(end * b);
        for &idx in &self.inner[bound as usize..r as usize] {
            unsafe {
                res.insert_unchecked(idx as usize);
            }
        }

        res
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
}

impl std::ops::Index<Range> for Indices {
    type Output = [Idx];

    fn index(&self, range: Range) -> &Self::Output {
        &self.inner[range.0 as usize..range.1 as usize]
    }
}

struct Stub {
    full: String,
    starts: Vec<Pos>,
    data: Vec<Key>,
}

fn iter_parts<'a>(full: &'a str, starts: &'a [Pos]) -> impl Iterator<Item = &'a str> + 'a {
    starts
        .iter()
        .circular_tuple_windows()
        .map(move |(&p1, &p2)| {
            if p1 < p2 {
                &full[p1 as usize..(p2 - 1) as usize]
            } else {
                &full[p1 as usize..]
            }
        })
}

impl Stub {
    fn iter(&self) -> impl Iterator<Item = Item<'_>> + '_ {
        let strs = iter_parts(&self.full, &self.starts);
        self.data.iter().copied().zip(strs)
    }
}

struct SA {
    sa: SuffixTable<'static, 'static>,
    data: Vec<Key>,   // len = V, number of strings
    starts: Vec<Pos>, // len = V
    indices: Indices,
}

#[cfg(feature = "sa-bench")]
static FLAGS: AtomicU32 = AtomicU32::new(0);

static STATS: AtomicU32 = AtomicU32::new(0);

impl SA {
    fn memory(&self) -> usize {
        let n = self.size();

        let r = n * (size_of::<Pos>() + size_of::<Idx>() + size_of::<u8>())
            + self.data.len()
                * (size_of::<Key>()
                    + size_of::<Pos>()
                    + self.indices.blocks.len() / size_of::<fixedbitset::Block>())
            + size_of::<Self>();

        #[cfg(feature = "sa-bench")]
        return r + n;
        #[cfg(not(feature = "sa-bench"))]
        r
    }

    fn stats(&self, dt: Duration) {
        let n = self.size();
        let v = self.len();
        let sa = (n * size_of::<Pos>()) >> 10;
        let blks = (self.indices.blocks.len() * v / size_of::<fixedbitset::Block>()) >> 10;
        let tot = self.memory() >> 10;
        let b = self.indices.block_size;
        let d = if v > 0 { n / v } else { 0 };
        debug!(
            "SA: sa=indices={sa}, blocks={blks}, total={tot} KiB (n={n}={v}*{d}, b={b}) in {dt:?}"
        );
    }

    fn from_inner(mut stub: Stub, t0: Instant) -> Self {
        stub.full.shrink_to_fit();
        stub.data.shrink_to_fit();
        stub.starts.shrink_to_fit();

        let t1 = Instant::now();
        let sa = SuffixTable::new(stub.full);
        debug!("SA table constructed in {:?}", t1.elapsed());

        let t1 = Instant::now();
        let indices = Indices::new(
            iter_parts(sa.text(), &stub.starts),
            stub.data.len() as Idx,
            sa.table(),
        );
        debug!("Indices constructed in {:?}", t1.elapsed());

        let dt = t0.elapsed();
        let r = SA {
            sa,
            data: stub.data,
            starts: stub.starts,
            indices,
        };

        if log_enabled!(log::Level::Debug) {
            r.stats(dt);
        }
        r
    }

    fn get_size_hint<S: AsRef<str>>(items: &[(Key, S)]) -> usize {
        items
            .iter()
            .map(|(_, s)| s.as_ref().len() + 1)
            .sum::<usize>()
    }

    fn from_iter<S: AsRef<str>, I: Iterator<Item = (Key, S)>>(iter: I) -> Self {
        let t0 = Instant::now();
        let items = iter.collect_vec();
        let mut full = String::with_capacity(Self::get_size_hint(&items).saturating_sub(1));

        let mut first = true;
        let (data, starts): (Vec<Key>, Vec<Pos>) = items
            .into_iter()
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

        SA::from_inner(Stub { full, starts, data }, t0)
    }

    fn extend<S: AsRef<str>, I: Iterator<Item = (Key, S)>>(self, iter: I) -> Self {
        if self.len() == 0 {
            return Self::from_iter(iter);
        }
        let t0 = Instant::now();
        let items = iter.collect_vec();
        let v = items.len();

        let mut inner = self.into_inner();
        inner.full.reserve(Self::get_size_hint(&items));
        inner.starts.reserve(v);
        inner.data.reserve(v);

        for (k, s) in items {
            // We have ensured `data` is non-empty, so we always put a separator here.
            inner.full.push('\\');
            let start = inner.full.len() as Pos;
            inner.full.push_str(s.as_ref());
            inner.starts.push(start);
            inner.data.push(k);
        }

        SA::from_inner(inner, t0)
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

    #[cfg(feature = "sa-inverted")]
    #[inline]
    fn indices(&self, pattern: &str) -> &[Idx] {
        &self.indices[self.indices_range(pattern)]
    }

    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn size(&self) -> usize {
        self.sa.len()
    }

    #[inline]
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

    #[inline]
    fn into_inner(self) -> Stub {
        let (full, _) = self.sa.into_parts();
        Stub {
            full: full.to_string(),
            starts: self.starts,
            data: self.data,
        }
    }

    fn single_select(&self, range: Range) -> impl Iterator<Item = Key> + '_ {
        self.indices[range]
            .iter()
            .sorted_unstable()
            .dedup()
            .copied()
            .map(move |idx| self.data[idx as usize])
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
            .copied()
            .filter_map(move |idx| {
                let (s, key) = self.resolve(idx);
                let l = s.len();
                if filters.iter().enumerate().all(|(p, tag)| {
                    p as Idx == min_idx || {
                        STATS.fetch_add(l as u32, Ordering::Relaxed);
                        s.contains(tag)
                    }
                } && !ban_filters.iter().any(|tag| {
                    STATS.fetch_add(l as u32, Ordering::Relaxed);
                    s.contains(tag)
                })) {
                    Some(key)
                } else {
                    None
                }
            })
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
        let indices = &self.indices;
        min_indices
            .iter()
            .sorted_unstable()
            .dedup()
            .copied()
            .filter_map(move |idx| {
                if all_ranges
                    .iter()
                    .enumerate()
                    .all(|(p, &range)| p as Idx == min_idx || indices.occured_in_range(idx, range))
                    && !ban_ranges
                        .iter()
                        .any(|&range| indices.occured_in_range(idx, range))
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
            STATS.fetch_add(range.len() as u32, Ordering::Relaxed);
            if p as Idx == min_idx {
                continue;
            }
            let indices = &self.indices[range];
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
            STATS.fetch_add(indices.len() as u32, Ordering::Relaxed);
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
        let ban_idxs = ban_filters
            .iter()
            .flat_map(|patt| self.indices(patt))
            .collect::<HashSet<_>>();
        STATS.fetch_add(ban_idxs.len() as u32, Ordering::Relaxed);
        self.data.iter().enumerate().filter_map(move |(idx, &key)| {
            if ban_idxs.contains(&(idx as Pos)) {
                None
            } else {
                Some(key)
            }
        })
    }

    /// Safety: `ranges` must yield at least one item.
    unsafe fn block_select(
        &self,
        mut ranges: impl Iterator<Item = Range>,
        ban_filters: &[String],
    ) -> impl Iterator<Item = Key> {
        let v = self.len();
        let first = unsafe { ranges.next().unwrap_unchecked() };
        let mut res = self.indices.join_blocks(first, v as Idx);
        for range in ranges {
            res.intersect_with(&self.indices.join_blocks(range, v as Idx));
        }
        for f in ban_filters {
            res.difference_with(&self.indices.join_blocks(self.indices_range(f), v as Idx));
        }
        res.ones()
            .map(|idx| self.data[idx])
            .collect_vec()
            .into_iter()
    }

    fn single_block_select(&self, range: Range) -> impl Iterator<Item = Key> {
        self.indices
            .join_blocks(range, self.len() as Idx)
            .ones()
            .map(|idx| self.data[idx])
            .collect_vec()
            .into_iter()
    }

    /// Safety: `ban_filters` must be non-empty.
    unsafe fn block_ban_select(&self, ban_filters: &[String]) -> impl Iterator<Item = Key> {
        let v: usize = self.len();
        // Safety: `IllustIndex::select` ensures at least one filter exists.
        let (first, rest) = unsafe { ban_filters.split_first().unwrap_unchecked() };
        let mut set = self
            .indices
            .join_blocks(self.indices_range(first), v as Idx);
        for f in rest {
            set.union_with(&self.indices.join_blocks(self.indices_range(f), v as Idx));
        }
        set.zeroes()
            .map(|idx| self.data[idx])
            .collect_vec()
            .into_iter()
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
            let indices = &self.indices[range];
            let t = indices.iter().copied().collect::<BTreeSet<_>>();
            STATS.fetch_add(t.len() as u32, Ordering::Relaxed);
            s = s.intersection(&t).copied().collect();
        }
        for range in ban_ranges {
            let indices = &self.indices[range];
            let t: BTreeSet<u32> = indices.iter().copied().collect::<BTreeSet<_>>();
            STATS.fetch_add(t.len() as u32, Ordering::Relaxed);
            s = s.difference(&t).copied().collect();
        }
        s.into_iter().map(|idx| self.data[idx as usize])
    }

    fn query_ranges(&self, patterns: &[String]) -> impl Iterator<Item = Range> {
        patterns.iter().map(|patt| self.indices_range(patt))
    }

    fn select<'a>(&'a self, query: Query<'a>) -> Box<dyn Iterator<Item = Key> + 'a> {
        if self.size() == 0 {
            return Box::new(std::iter::empty());
        }

        let filters = query.filters;
        let ban_filters = query.ban_filters;

        #[cfg(feature = "sa-bench")]
        let flags = FLAGS.load(Ordering::Relaxed) & 0xf;

        #[cfg(feature = "sa-bench")]
        if flags == 0x9 {
            debug!("forced brute select");
            return Box::new(
                self.brute_select(self.query_ranges(filters), self.query_ranges(ban_filters)),
            );
        }

        match filters.len() {
            0 => {
                #[cfg(feature = "sa-bench")]
                if flags == 0x5 {
                    debug!("forced inverted_ban_select");
                    return Box::new(self.inverted_ban_select(ban_filters));
                }
                // Safety: `Query::new` ensures at least one filter exists.
                // Since `filters.len() == 0`, `ban_filters` must be non-empty.
                return Box::new(unsafe { self.block_ban_select(ban_filters) });
            }
            1 => {
                #[cfg(feature = "sa-bench")]
                let cond = flags == 0;
                #[cfg(not(feature = "sa-bench"))]
                let cond = true;
                if cond && ban_filters.is_empty() {
                    let range = self.indices_range(&filters[0]);
                    if range.len() >= 3000 {
                        return Box::new(self.single_block_select(range));
                    }
                    return Box::new(self.single_select(range));
                }
            }
            _ => {}
        }

        let all = self.query_ranges(filters);

        #[cfg(feature = "sa-bench")]
        if flags == 0x7 {
            debug!("forced block_select");
            // Safety: we have ensured `filters.len() >= 1`.
            return Box::new(unsafe { self.block_select(all, ban_filters) });
        }

        let all = all.collect_vec();

        // Safety: we have ensured `filters.len() >= 1`.
        let (min_idx, &min) = unsafe {
            all.iter()
                .enumerate()
                .min_by_key(|(_, poses)| poses.len())
                .unwrap_unchecked()
        };

        let min_idx = min_idx as Idx;
        let min_len = min.len();

        if min_len == 0 {
            return Box::new(std::iter::empty());
        }

        if log_enabled!(log::Level::Debug) {
            let lens = all.iter().map(RangeExt::len).collect_vec();
            let sum_len: usize = lens.iter().sum();
            let c = min_len as f64 / sum_len as f64;
            debug!("indices lengths: {lens:?}, c = {c:.6}");
        }

        let min_indices = &self.indices[min];

        #[cfg(feature = "sa-bench")]
        if flags == 0x1 {
            debug!("forced heuristic_select");
            return Box::new(self.heuristic_select(min_idx, min_indices, filters, ban_filters));
        }

        #[cfg(feature = "sa-bench")]
        if flags == 0x5 {
            debug!("forced inverted_select");
            return self.inverted_select(all, min_idx, min_indices, ban_filters);
        }

        #[cfg(all(feature = "sa-bench", feature = "sa-inverted"))]
        if flags == 0x3 {
            debug!("forced binary_select");
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

        // Safety: we have ensured `filters.len() >= 1`.
        Box::new(unsafe { self.block_select(all.into_iter(), ban_filters) })
    }
}

impl Default for SA {
    fn default() -> Self {
        Self::from_iter::<&str, _>(std::iter::empty())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Query<'a> {
    filters: &'a [String],
    ban_filters: &'a [String],
}

impl<'a> Query<'a> {
    pub fn new(filters: &'a [String], ban_filters: &'a [String]) -> Self {
        assert!(!(filters.is_empty() && ban_filters.is_empty()));
        Query {
            filters,
            ban_filters,
        }
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
    major: SA,
    minor: Option<SA>,
    dedup: bool,
}

impl Index {
    pub fn new<S: AsRef<str>, I: Iterator<Item = (Key, S)>>(data: I) -> Self {
        Self {
            major: SA::from_iter(data),
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
                    SA::from_iter(combined.into_iter())
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

                    SA::from_iter(combined.into_iter())
                } else {
                    minor.extend(data)
                });
            }
        } else {
            self.minor = Some(SA::from_iter(data.into_iter()));
        }
    }
}
