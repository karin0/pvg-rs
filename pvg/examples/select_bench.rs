//! Replays select queries against a favorites database, timing the search
//! index and a linear `str::contains` scan over the same intros, and checks
//! that both find the same illusts. Each stdin line is a JSON pair of
//! normalized filters and ban filters, `[["a", "b"], ["c"]]`. A pair of empty
//! lists selects every illust without the index, and is skipped.
//!
//! Usage: `select_bench DB_FILE < queries.jsonl`. Opening the database applies
//! the schema, so `DB_FILE` should be a copy.

use anyhow::{Context, Result, bail, ensure};
use pvg::model::IllustIndex;
use pvg::search::{Index, Query};
use std::io::BufRead;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const RUNS: usize = 5;

type Intros = [(pixiv::IllustId, String)];

fn scan(intros: &Intros, filters: &[String], bans: &[String]) -> Vec<pixiv::IllustId> {
    intros
        .iter()
        .filter(|(_, s)| {
            filters.iter().all(|f| s.contains(f.as_str()))
                && !bans.iter().any(|b| s.contains(b.as_str()))
        })
        .map(|(id, _)| *id)
        .collect()
}

/// The median of `RUNS` timed calls, and the result of the last one.
fn time<T>(mut f: impl FnMut() -> T) -> (Duration, T) {
    let mut times = Vec::with_capacity(RUNS);
    let mut res = None;
    for _ in 0..RUNS {
        let t0 = Instant::now();
        res = Some(f());
        times.push(t0.elapsed());
    }
    times.sort_unstable();
    (times[RUNS / 2], res.unwrap())
}

fn quantiles(mut v: Vec<Duration>) -> String {
    v.sort_unstable();
    let q = |percent: usize| v[(v.len() * percent / 100).min(v.len() - 1)];
    format!(
        "p50 {:?}, p90 {:?}, p99 {:?}, max {:?}",
        q(50),
        q(90),
        q(99),
        v[v.len() - 1]
    )
}

#[tokio::main]
async fn main() -> Result<()> {
    let db: PathBuf = std::env::args()
        .nth(1)
        .context("usage: select_bench DB_FILE < queries.jsonl")?
        .into();
    let intros = IllustIndex::connect(&db, true, false).await?.intros();
    let size: usize = intros.iter().map(|(_, s)| s.len()).sum();

    let t0 = Instant::now();
    let index = Index::new(intros.iter().map(|(id, s)| (*id, s.as_str())));
    println!(
        "{} intros, {} KiB; index built in {:?}, {} MiB",
        intros.len(),
        size >> 10,
        t0.elapsed(),
        index.memory() >> 20
    );

    let (mut by_index, mut by_scan) = (Vec::new(), Vec::new());
    for line in std::io::stdin().lock().lines() {
        let (filters, bans): (Vec<String>, Vec<String>) = serde_json::from_str(&line?)?;
        if filters.is_empty() && bans.is_empty() {
            continue;
        }
        let (di, mut a) = time(|| index.select(Query::new(&filters, &bans)));
        let (ds, mut b) = time(|| scan(&intros, &filters, &bans));
        a.sort_unstable();
        b.sort_unstable();
        if a != b {
            let only = |x: &[pixiv::IllustId], y: &[pixiv::IllustId]| {
                x.iter()
                    .filter(|id| y.binary_search(id).is_err())
                    .take(5)
                    .copied()
                    .collect::<Vec<_>>()
            };
            bail!(
                "{filters:?} - {bans:?}: index finds {}, scan {}; only index {:?}, only scan {:?}",
                a.len(),
                b.len(),
                only(&a, &b),
                only(&b, &a)
            );
        }
        println!(
            "{filters:?} - {bans:?}: {} illusts, index {di:?}, scan {ds:?}",
            a.len()
        );
        by_index.push(di);
        by_scan.push(ds);
    }
    ensure!(!by_index.is_empty(), "no queries on stdin");
    println!("{} queries", by_index.len());
    println!("index: {}", quantiles(by_index));
    println!("scan:  {}", quantiles(by_scan));
    Ok(())
}
