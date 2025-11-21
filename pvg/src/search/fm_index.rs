use super::types::Pos;
use bio::alphabets::Alphabet;
use bio::data_structures::bwt::{BWT, less};
use bio::data_structures::fmindex::FMIndexable;
use itertools::Itertools;
use std::mem::size_of;

const OCC_K: Pos = 512;

pub struct Occ {
    occ: Vec<Vec<Pos>>,
}

type BWTSlice = [u8];

// https://github.com/rust-bio/rust-bio/blob/14c04251e2bd469ccb706bc77e4107cfc58295f4/src/data_structures/bwt.rs
impl Occ {
    fn memory(&self) -> usize {
        self.occ
            .iter()
            .map(|v| v.len() * size_of::<Pos>())
            .sum::<usize>()
            + size_of::<Self>()
    }

    pub fn new(bwt: &BWTSlice, alphabet: &Alphabet) -> Self {
        let n = bwt.len();
        let m = alphabet
            .max_symbol()
            .expect("Expecting non-empty alphabet.") as usize
            + 1;
        let mut alpha = alphabet.symbols.iter().collect::<Vec<usize>>();

        if (b'$' as usize) < m && !alphabet.is_word(b"$") {
            alpha.push(b'$' as usize);
        }
        let mut occ = vec![Vec::new(); m];
        let mut curr_occ = vec![0u32; m];

        for &a in &alpha {
            occ[a].reserve(n / OCC_K as usize);
        }

        for (i, &c) in bwt.iter().enumerate() {
            curr_occ[c as usize] += 1;

            if i % OCC_K as usize == 0 {
                for &a in &alpha {
                    occ[a].push(curr_occ[a]);
                }
            }
        }

        for a in &mut occ {
            a.shrink_to_fit();
        }
        occ.shrink_to_fit();

        Occ { occ }
    }

    pub fn get(&self, bwt: &BWTSlice, r: usize, a: u8) -> usize {
        let lo_checkpoint = r / OCC_K as usize;
        let lo_occ = self.occ[a as usize][lo_checkpoint] as usize;

        if OCC_K > 64 {
            let hi_checkpoint = lo_checkpoint + 1;
            if let Some(&hi_occ) = self.occ[a as usize].get(hi_checkpoint) {
                if lo_occ == hi_occ as usize {
                    return lo_occ;
                }

                let hi_idx = hi_checkpoint * OCC_K as usize;
                if (hi_idx - r) < (OCC_K as usize / 2) {
                    return hi_occ as usize - bytecount::count(&bwt[r + 1..=hi_idx], a);
                }
            }
        }

        let lo_idx = lo_checkpoint * OCC_K as usize;
        bytecount::count(&bwt[lo_idx + 1..=r], a) + lo_occ
    }
}

pub struct FMIndex {
    occ: Occ,
    bwt: BWT,
    less: Vec<Pos>,
}

impl FMIndexable for FMIndex {
    fn occ(&self, r: usize, a: u8) -> usize {
        self.occ.get(&self.bwt, r, a)
    }

    fn less(&self, a: u8) -> usize {
        self.less[a as usize] as usize
    }

    fn bwt(&self) -> &BWT {
        &self.bwt
    }
}

impl FMIndex {
    pub fn memory(&self) -> usize {
        let bwt = self.bwt.len();
        let less = self.less.len() * size_of::<Pos>();
        let occ = self.occ.memory();
        bwt + less + occ
    }

    fn stats(&self, sigma: usize) {
        let bwt = self.bwt.len();
        let less = self.less.len() * size_of::<Pos>();
        let occ = self.occ.memory();
        let tot = bwt + less + occ;
        debug!(
            "FM: bwt={}, less={}, occ={}, total={} MiB, sigma={sigma}",
            bwt >> 20,
            less >> 20,
            occ >> 20,
            tot >> 20,
        );
    }

    pub fn new(text: &[u8], mut bwt: BWT) -> Self {
        bwt.shrink_to_fit();

        let alphabet = Alphabet::new(text);

        let occ = Occ::new(&bwt, &alphabet);
        let mut less = less(text, &alphabet)
            .into_iter()
            .map(|x| x as Pos)
            .collect_vec();
        less.shrink_to_fit();

        let r = FMIndex { occ, bwt, less };
        if log_enabled!(log::Level::Debug) {
            r.stats(alphabet.len());
        }
        r
    }
}
