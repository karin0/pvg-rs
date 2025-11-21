use super::types::{Idx, Pos, STATS};
use fixedbitset::FixedBitSet;
use integer_sqrt::IntegerSquareRoot;
use itertools::Itertools;
use std::sync::atomic::Ordering;

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
    inner: Vec<Idx>, // len = N, total length of all strings (in bytes)
    #[cfg(feature = "sa-inverted")]
    occurrences: Vec<Vec<Pos>>, // len = V, but total size of inner vecs = N
    blocks: Vec<FixedBitSet>, // len ~ sqrt(N)
    pub block_size: Pos,
}

impl Indices {
    pub fn memory(&self, v: usize) -> (usize, usize) {
        let n = self.inner.len();
        let blocks = self.blocks.len() * v / size_of::<fixedbitset::Block>();
        let r = n * size_of::<Idx>() + blocks + size_of::<Self>();

        #[cfg(feature = "sa-inverted")]
        let r = r + n;

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

        Indices {
            inner: indices,
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
    pub fn occured_in_range(&self, idx: Idx, range: Range) -> bool {
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
