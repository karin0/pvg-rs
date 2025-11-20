use anyhow::Result;
use itertools::Itertools;
use lz4::block::{compress_bound, compress_to_buffer, decompress};
use pixiv::IllustId;
use sqlx::{
    QueryBuilder, SqlitePool, query, query_file, query_scalar, sqlite::SqliteConnectOptions,
};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct Store {
    pool: SqlitePool,
}

const BATCH_SIZE: usize = 1000;

fn to_blob(json: &[u8]) -> std::io::Result<Vec<u8>> {
    if json[0] == b'L' {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "already compressed",
        ));
    }

    // 4 bytes for lz4 src size, 1 byte for 'L'
    let size = 5 + compress_bound(json.len())?;
    let mut buf = vec![0; size];
    buf[0] = b'L';
    let size = 1 + compress_to_buffer(json, None, true, &mut buf[1..])?;
    buf.resize(size, 0);
    Ok(buf)
}

static UNCOMPRESSED_CNT: AtomicUsize = AtomicUsize::new(0);
static DECOMPRESS_COST: AtomicUsize = AtomicUsize::new(0);

fn to_json(blob: Vec<u8>) -> std::io::Result<Vec<u8>> {
    if blob[0] != b'L' {
        let c = UNCOMPRESSED_CNT.fetch_add(1, Ordering::Relaxed);
        if c < 10 {
            warn!(
                "uncompressed blob: {}",
                std::str::from_utf8(&blob).unwrap_or("<non-utf8>")
            );
        }
        return Ok(blob);
    }

    let t0 = Instant::now();
    let r = decompress(&blob[1..], None)?;
    let cost = t0.elapsed().as_nanos() as usize;
    DECOMPRESS_COST.fetch_add(cost, Ordering::Relaxed);
    Ok(r)
}

impl Store {
    pub async fn open(db_path: &Path, create: bool) -> Result<Self> {
        let opts = SqliteConnectOptions::new()
            .filename(db_path)
            .create_if_missing(create);

        let pool = SqlitePool::connect_with(opts).await?;
        query_file!("../schema.sql").execute(&pool).await?;
        Ok(Store { pool })
    }

    pub async fn illusts(&self) -> Result<Vec<Vec<u8>>> {
        let r = query_scalar!("SELECT data FROM Illust ORDER BY id")
            .fetch_all(&self.pool)
            .await?
            .into_iter()
            .map(to_json)
            .collect::<std::io::Result<Vec<_>>>()?;
        let cnt = UNCOMPRESSED_CNT.swap(0, Ordering::Relaxed);
        if cnt > 0 {
            warn!("found {cnt} uncompressed blobs!");
        }
        let time = DECOMPRESS_COST.swap(0, Ordering::Relaxed);
        info!("decompression took {:?}", Duration::from_nanos(time as u64));
        Ok(r)
    }

    async fn _upsert<'a, I>(&self, illusts: I) -> Result<()>
    where
        I: IntoIterator<Item = (&'a Vec<u8>, IllustId)>,
    {
        let t0 = tokio::time::Instant::now();
        let mut query = QueryBuilder::new("INSERT INTO Illust (data, iid)");

        query
            .push_values(
                illusts
                    .into_iter()
                    .map(|(data, iid)| Ok((to_blob(data)?, iid)))
                    .collect::<std::io::Result<Vec<_>>>()?,
                |mut b, (blob, iid)| {
                    b.push_bind(blob).push_bind(iid);
                },
            )
            .push(" ON CONFLICT(iid) DO UPDATE SET data = excluded.data, iid = excluded.iid");

        let r = query.build().execute(&self.pool).await?;
        if log_enabled!(log::Level::Debug) {
            debug!(
                "upsert took {:?}: affected={:?} rowid={}",
                t0.elapsed(),
                r.rows_affected(),
                r.last_insert_rowid()
            );
        }
        Ok(())
    }

    pub async fn upsert<'a, I>(&self, illusts: I) -> Result<()>
    where
        I: Iterator<Item = (&'a Vec<u8>, IllustId)>,
    {
        let all = illusts.collect_vec();
        if all.len() <= BATCH_SIZE {
            return self._upsert(all).await;
        }

        let tx = self.pool.begin().await?;
        for chunk in all.chunks(BATCH_SIZE) {
            self._upsert(chunk.iter().copied()).await?;
        }
        Ok(tx.commit().await?)
    }

    async fn _overwrite<'a, I>(&self, illusts: I) -> Result<()>
    where
        I: IntoIterator<Item = (&'a Vec<u8>, IllustId)>,
    {
        let t0 = tokio::time::Instant::now();
        let mut query = QueryBuilder::new("INSERT INTO Illust (data, iid)");

        query.push_values(
            illusts
                .into_iter()
                .map(|(data, iid)| Ok((to_blob(data)?, iid)))
                .collect::<std::io::Result<Vec<_>>>()?,
            |mut b, (data, iid)| {
                b.push_bind(data).push_bind(iid);
            },
        );

        let r = query.build().execute(&self.pool).await?;
        info!("overwrite took {:?}: {:?}", t0.elapsed(), r);
        Ok(())
    }

    pub async fn overwrite<'a, I>(&self, illusts: I) -> Result<()>
    where
        I: Iterator<Item = (&'a Vec<u8>, IllustId)>,
    {
        let all = illusts.collect_vec();
        warn!("OVERWRITING WITH {} ILLUSTS!", all.len());

        let tx = self.pool.begin().await?;
        query!("DROP TABLE Illust").execute(&self.pool).await?;
        query_file!("../schema.sql").execute(&self.pool).await?;
        query!("VACUUM").execute(&self.pool).await?;

        if all.len() <= BATCH_SIZE {
            self._overwrite(all).await?;
        } else {
            for chunk in all.chunks(BATCH_SIZE) {
                self._overwrite(chunk.iter().copied()).await?;
            }
        }
        Ok(tx.commit().await?)
    }
}
