use super::types::{Idx, Pos, STATS};
use fixedbitset::FixedBitSet;
use integer_sqrt::IntegerSquareRoot;
use itertools::Itertools;
use std::sync::atomic::Ordering;

#[cfg(feature = "sa-packed")]
use packedvec::PackedVec;

pub type Range = (Pos, Pos);

pub trait RangeExt {
    fn len(&self) -> usize;
}

impl RangeExt for Range {
    #[inline]
    fn len(&self) -> usize {
        (self.1 - self.0) as usize
    }
}

pub struct Indices {
    #[cfg(any(feature = "sa-packed-bench", not(feature = "sa-packed")))]
    inner: Vec<Idx>, // len = N, total length of all strings (in bytes)
    #[cfg(feature = "sa-packed")]
    packed_inner: PackedVec<Idx>, // packed variant
    #[cfg(feature = "sa-inverted")]
    occurrences: Vec<Vec<Pos>>, // len = V, but total size of inner vecs = N
    blocks: Vec<FixedBitSet>, // len ~ sqrt(N)
    pub block_size: Pos,
}

impl Indices {
    pub fn memory(&self, v: usize) -> (usize, usize) {
        #[cfg(not(feature = "sa-packed"))]
        let n = self.inner.len();

        #[cfg(feature = "sa-packed")]
        let n = self.packed_inner.len();

        let blocks = self.blocks.len() * v / size_of::<fixedbitset::Block>();

        let r = blocks + size_of::<Self>();

        #[cfg(any(feature = "sa-packed-bench", not(feature = "sa-packed")))]
        let r = r + n * size_of::<Idx>();

        #[cfg(feature = "sa-packed")]
        let r = r + n * self.packed_inner.bwidth() / 8;

        #[cfg(feature = "sa-inverted")]
        let r = r + n * size_of::<Pos>();

        (r, blocks)
    }

    pub fn new<S: AsRef<str>, I: Iterator<Item = S>>(iter: I, v: Idx, sa: &[usize]) -> Indices {
        let raw_indices = iter
            .enumerate()
            .flat_map(|(idx, s)| {
                std::iter::repeat_n(idx as Pos, s.as_ref().len() + usize::from(idx != 0))
            })
            .collect_vec();

        let mut indices = sa.iter().map(|&p| raw_indices[p]).collect_vec();
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

        #[cfg(feature = "sa-packed")]
        let packed_inner = {
            #[cfg(feature = "sa-packed-bench")]
            let indices = indices.clone();

            let packed = PackedVec::new(indices);
            let n = packed.len();
            let width = packed.bwidth();
            let size = (n * width) >> 23;
            debug!("packed indices: {n} * {width} bits = {size} MiB");
            packed
        };

        Indices {
            #[cfg(any(feature = "sa-packed-bench", not(feature = "sa-packed")))]
            inner: indices,
            #[cfg(feature = "sa-packed")]
            packed_inner,
            #[cfg(feature = "sa-inverted")]
            occurrences,
            blocks,
            block_size,
        }
    }

    #[inline]
    pub fn join_blocks(&self, range: Range, v: Idx) -> FixedBitSet {
        let (l, r) = range;
        let b = self.block_size;
        let start = l.div_ceil(b);
        let end = r / b;

        let bound = r.min(start * b);
        if r <= bound {
            // Same block, r == bound.
            let mut res = FixedBitSet::with_capacity(v as usize);
            for idx in self.slice((l, r)) {
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

        for idx in self.slice((l, bound)) {
            unsafe {
                res.insert_unchecked(idx as usize);
            }
        }

        let bound = l.max(end * b);
        for idx in self.slice((bound, r)) {
            unsafe {
                res.insert_unchecked(idx as usize);
            }
        }

        res
    }

    #[cfg(feature = "sa-inverted")]
    pub fn occured_in_range(&self, idx: Idx, range: Range) -> bool {
        let occurences = &self.occurrences[idx as usize];
        if let Err(bound) = occurences.binary_search(&range.0) {
            occurences.get(bound).is_some_and(|&p| p < range.1)
        } else {
            true
        }
    }

    #[cfg(any(feature = "sa-packed-bench", not(feature = "sa-packed")))]
    #[inline]
    fn slice_raw(&self, range: Range) -> impl Iterator<Item = Idx> + '_ {
        self.inner[range.0 as usize..range.1 as usize]
            .iter()
            .copied()
    }

    #[cfg(feature = "sa-packed")]
    #[inline]
    fn slice_packed(&self, range: Range) -> impl Iterator<Item = Idx> + Clone + '_ {
        IndicesSlice {
            indices: &self.packed_inner,
            pos: range.0,
            end: range.1.min(self.packed_inner.len() as Pos),
        }
    }

    #[inline]
    pub fn slice(&self, range: Range) -> impl Iterator<Item = Idx> + '_ {
        #[cfg(any(feature = "sa-packed-bench", not(feature = "sa-packed")))]
        let r1 = self.slice_raw(range);

        #[cfg(feature = "sa-packed")]
        let r2 = self.slice_packed(range);

        #[cfg(feature = "sa-packed-bench")]
        return {
            use std::time::Instant;

            let r = r2.clone();
            let t0 = Instant::now();
            let vec = r1.collect_vec();
            let dt = t0.elapsed();
            let t0 = Instant::now();
            let vec2 = r2.collect_vec();
            let dt2 = t0.elapsed();
            debug!(
                "Indices{range:?}: slice {dt:?} vs packed {dt2:?} ({:.3}x) (len={})",
                dt2.as_nanos().max(1) as f64 / dt.as_nanos().max(1) as f64,
                vec.len()
            );
            assert_eq!(vec, vec2);
            r
        };

        #[cfg(all(feature = "sa-packed", not(feature = "sa-packed-bench")))]
        return r2;

        #[cfg(all(not(feature = "sa-packed"), not(feature = "sa-packed-bench")))]
        r1
    }
}

#[cfg(feature = "sa-packed")]
#[derive(Debug, Clone)]
struct IndicesSlice<'a> {
    indices: &'a PackedVec<Idx>,
    pos: Pos,
    end: Pos,
}

#[cfg(feature = "sa-packed")]
impl Iterator for IndicesSlice<'_> {
    type Item = Idx;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.end {
            let pos = self.pos as usize;
            self.pos += 1;
            // Safety: pos < end <= indices.len()
            Some(unsafe { self.indices.get_unchecked(pos) })
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = (self.end - self.pos) as usize;
        (len, Some(len))
    }
}
