use bio::data_structures::suffix_array::suffix_array;
use itertools::Itertools;
use std::mem::size_of;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

#[cfg(feature = "sa-bench")]
use std::sync::atomic::AtomicU32;

#[cfg(feature = "fm-bench")]
use std::sync::atomic::AtomicU64;

use super::indices::{Indices, Range, RangeExt};
use super::types::{Idx, Item, Key, Pos, STATS};

#[cfg(feature = "fm-index")]
use super::fm_index::FMIndex;

// https://github.com/BurntSushi/suffix/blob/5ba4f72941872b697ff3c216f8315ff6de4bf5d7/src/table.rs
#[cfg(any(feature = "fm-bench", not(feature = "fm-index")))]
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

pub struct Stub {
    full: String,     // len = N
    starts: Vec<Pos>, // len = V, number of strings
    data: Vec<Key>,   // len = V
}

impl Stub {
    fn memory(&self) -> usize {
        self.full.len() * size_of::<u8>()
            + self.starts.len() * size_of::<Pos>()
            + self.data.len() * size_of::<Key>()
            + size_of::<Self>()
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = Item<'_>> + '_ {
        let strs = iter_parts(&self.full, &self.starts);
        self.data.iter().copied().zip(strs)
    }

    #[inline]
    fn resolve(&self, idx: u32) -> (&str, Key) {
        let idx = idx as usize;
        let starts = &self.starts;
        let full = &self.full;
        (
            if idx + 1 < self.data.len() {
                &full[starts[idx] as usize..(starts[idx + 1] - 1) as usize]
            } else {
                &full[starts[idx] as usize..]
            },
            self.data[idx],
        )
    }
}

#[cfg(feature = "sa-bench")]
pub static FLAGS: AtomicU32 = AtomicU32::new(0);

#[cfg(feature = "fm-bench")]
static PERF_SA: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "fm-bench")]
static PERF_FM: AtomicU64 = AtomicU64::new(0);

pub struct Segment {
    #[cfg(any(feature = "fm-bench", not(feature = "fm-index")))]
    sa: Vec<Pos>, // len = N
    #[cfg(feature = "fm-index")]
    fm: FMIndex,
    inner: Stub,
    indices: Indices,
}

impl Segment {
    pub fn memory(&self) -> usize {
        let mut r = self.inner.memory() + self.indices.memory(self.len()).0 + size_of::<Self>();

        #[cfg(any(feature = "fm-bench", not(feature = "fm-index")))]
        {
            r += self.sa.len() * size_of::<Pos>();
        }

        #[cfg(feature = "fm-index")]
        {
            r += self.fm.memory();
        }

        r
    }

    fn stats(&self, dt: Duration) {
        let n = self.size();
        let v = self.len();

        let (indices, blocks) = self.indices.memory(v);
        let indices = (indices - blocks) >> 20;
        let blocks = blocks >> 20;

        #[cfg(any(feature = "fm-bench", not(feature = "fm-index")))]
        let sa = (self.sa.len() * size_of::<Pos>()) >> 20;
        #[cfg(not(any(feature = "fm-bench", not(feature = "fm-index"))))]
        let sa = 0;

        #[cfg(feature = "fm-index")]
        let fm = self.fm.memory() >> 20;

        #[cfg(not(feature = "fm-index"))]
        let fm = 0;

        let tot = self.memory() >> 20;

        let b = self.indices.block_size;
        let d = if v > 0 { n / v } else { 0 };
        debug!(
            "SA: n={n}={v}*{d}, sa={sa}, fm={fm}, indices={indices}, blocks={blocks} ({b}), total={tot} MiB in {dt:?}"
        );
    }

    fn from_inner(mut stub: Stub, t0: Instant) -> Self {
        stub.data.shrink_to_fit();
        stub.starts.shrink_to_fit();

        let n = stub.full.len();

        stub.full.push('\0');
        let text = stub.full.as_bytes();

        let t1 = Instant::now();
        let sa = suffix_array(text);
        debug!("SA constructed in {:?}", t1.elapsed());

        let table = &sa[1..];
        assert_eq!(table.len(), n);

        let t1 = Instant::now();
        let indices = Indices::new(
            iter_parts(&stub.full, &stub.starts),
            stub.data.len() as Idx,
            table,
        );
        debug!("Indices constructed in {:?}", t1.elapsed());

        #[cfg(feature = "fm-index")]
        let fm = {
            use bio::data_structures::bwt::bwt;

            let t1 = Instant::now();
            let bwt = bwt(text, &sa);
            #[cfg(not(feature = "fm-bench"))]
            drop(sa);
            let r = FMIndex::new(text, bwt);
            debug!("FMIndex constructed in {:?}", t1.elapsed());
            r
        };

        #[cfg(any(feature = "fm-bench", not(feature = "fm-index")))]
        let sa = table.iter().copied().map(|x| x as Pos).collect_vec();

        stub.full.pop();
        stub.full.shrink_to_fit();

        let dt = t0.elapsed();
        let r = Segment {
            #[cfg(any(feature = "fm-bench", not(feature = "fm-index")))]
            sa,
            #[cfg(feature = "fm-index")]
            fm,
            inner: stub,
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
            + 1
    }

    pub fn from_iter<S: AsRef<str>, I: Iterator<Item = (Key, S)>>(iter: I) -> Self {
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

        Segment::from_inner(Stub { full, starts, data }, t0)
    }

    pub fn extend<S: AsRef<str>, I: Iterator<Item = (Key, S)>>(self, iter: I) -> Self {
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

        Segment::from_inner(inner, t0)
    }

    #[cfg(any(feature = "fm-bench", not(feature = "fm-index")))]
    fn suffix_bytes(&self, i: usize) -> &[u8] {
        &self.inner.full.as_bytes()[self.sa[i] as usize..]
    }

    // https://github.com/BurntSushi/suffix/blob/5ba4f72941872b697ff3c216f8315ff6de4bf5d7/src/table.rs
    #[cfg(any(feature = "fm-bench", not(feature = "fm-index")))]
    fn indices_range_sa(&self, query: &str) -> Range {
        #[cfg(feature = "fm-bench")]
        let t0 = Instant::now();

        let (text, query) = (self.inner.full.as_bytes(), query.as_bytes());
        if text.is_empty()
            || query.is_empty()
            || (query < self.suffix_bytes(0) && !self.suffix_bytes(0).starts_with(query))
            || query > self.suffix_bytes(text.len() - 1)
        {
            return (0, 0);
        }

        let table = &self.sa;
        let start = binary_search(table, |&sufi| query <= &text[sufi as usize..]);
        let end = start
            + binary_search(&table[start..], |&sufi| {
                !text[sufi as usize..].starts_with(query)
            });

        let r = if start >= end {
            (0, 0)
        } else {
            (start as Pos, end as Pos)
        };

        #[cfg(feature = "fm-bench")]
        {
            let dt = t0.elapsed();
            PERF_SA.fetch_add(dt.as_nanos() as u64, Ordering::Relaxed);
        }
        r
    }

    #[cfg(feature = "fm-index")]
    fn indices_range_fm(&self, pattern: &str) -> Range {
        use bio::data_structures::fmindex::{BackwardSearchResult, FMIndexable};

        #[cfg(feature = "fm-bench")]
        let t0 = Instant::now();

        let interval = self.fm.backward_search(pattern.as_bytes().iter());
        let r = if let BackwardSearchResult::Complete(interval) = interval {
            ((interval.lower - 1) as Pos, (interval.upper - 1) as Pos)
        } else {
            (0, 0)
        };

        #[cfg(feature = "fm-bench")]
        {
            let dt = t0.elapsed();
            PERF_FM.fetch_add(dt.as_nanos() as u64, Ordering::Relaxed);
        }
        r
    }

    #[cfg(feature = "fm-bench")]
    fn indices_range(&self, pattern: &str) -> Range {
        let r = self.indices_range_sa(pattern);
        let r2 = self.indices_range_fm(pattern);
        if r != r2 {
            warn!("SA and FM ranges differ for pattern '{pattern}': SA={r:?}, FM={r2:?}");
        }
        r
    }

    #[cfg(not(any(feature = "fm-bench", feature = "fm-index")))]
    fn indices_range(&self, pattern: &str) -> Range {
        self.indices_range_sa(pattern)
    }

    #[cfg(all(not(feature = "fm-bench"), feature = "fm-index"))]
    fn indices_range(&self, pattern: &str) -> Range {
        self.indices_range_fm(pattern)
    }

    #[cfg(feature = "sa-bench")]
    #[inline]
    fn indices(&self, pattern: &str) -> &[Idx] {
        &self.indices[self.indices_range(pattern)]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.data.len()
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.inner.full.len()
    }

    #[inline]
    pub fn into_inner(self) -> Stub {
        self.inner
    }

    fn single_select(&self, range: Range) -> impl Iterator<Item = Key> + '_ {
        self.indices[range]
            .iter()
            .sorted_unstable()
            .dedup()
            .copied()
            .map(move |idx| self.inner.data[idx as usize])
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
                let (s, key) = self.inner.resolve(idx);
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
                    Some(self.inner.data[idx as usize])
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
                .map(move |idx| self.inner.data[idx as usize]),
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
        self.inner
            .data
            .iter()
            .enumerate()
            .filter_map(move |(idx, &key)| {
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
            .map(|idx| self.inner.data[idx])
            .collect_vec()
            .into_iter()
    }

    fn single_block_select(&self, range: Range) -> impl Iterator<Item = Key> {
        self.indices
            .join_blocks(range, self.len() as Idx)
            .ones()
            .map(|idx| self.inner.data[idx])
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
            .map(|idx| self.inner.data[idx])
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
        s.into_iter().map(|idx| self.inner.data[idx as usize])
    }

    fn query_ranges(&self, patterns: &[String]) -> impl Iterator<Item = Range> {
        patterns.iter().map(|patt| self.indices_range(patt))
    }

    fn do_select<'a>(&'a self, query: Query<'a>) -> Box<dyn Iterator<Item = Key> + 'a> {
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

    #[cfg(feature = "fm-bench")]
    pub fn select<'a>(&'a self, query: Query<'a>) -> Box<dyn Iterator<Item = Key> + 'a> {
        PERF_SA.store(0, Ordering::Relaxed);
        PERF_FM.store(0, Ordering::Relaxed);
        let r = self.do_select(query);

        let d1 = Duration::from_nanos(PERF_SA.load(Ordering::Relaxed));
        let d2 = Duration::from_nanos(PERF_FM.load(Ordering::Relaxed));
        debug!("SA time: {d1:?}, FM time: {d2:?}");
        r
    }

    #[cfg(not(feature = "fm-bench"))]
    pub fn select<'a>(&'a self, query: Query<'a>) -> Box<dyn Iterator<Item = Key> + 'a> {
        self.do_select(query)
    }
}

impl Default for Segment {
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
