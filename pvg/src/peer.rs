use crate::config::PeerConfig;
use anyhow::Result;
use pixiv::IllustId;
use pixiv::model::{MetaPage, MetaSinglePage};
use serde::Deserialize;
use sqlx::SqlitePool;
use sqlx::sqlite::SqliteConnectOptions;
use std::path::PathBuf;
use tokio::fs;

/// Another archive of the same pixiv illusts, whose originals are copied
/// before any download. Its database has a table `Illust(id, data)`, `data`
/// being the raw illust JSON, and its directory holds the p0 original of each
/// row's current URL under the URL basename. A re-upload can keep the
/// basename, so a page is taken only when its URL equals the peer's p0 URL.
#[derive(Debug)]
pub struct Peer {
    pool: SqlitePool,
    pix_dir: PathBuf,
}

#[derive(Deserialize)]
struct Pages {
    meta_single_page: MetaSinglePage,
    meta_pages: Vec<MetaPage>,
}

impl Pages {
    fn p0(self) -> Option<String> {
        self.meta_pages
            .into_iter()
            .next()
            .map(|p| p.image_urls.original)
            .or(self.meta_single_page.original_image_url)
    }
}

impl Peer {
    pub async fn open(conf: PeerConfig) -> Result<Self> {
        let opts = SqliteConnectOptions::new()
            .filename(&conf.db)
            .read_only(true);
        Ok(Self {
            pool: SqlitePool::connect_with(opts).await?,
            pix_dir: conf.pix_dir,
        })
    }

    /// The peer's file of the page at `url`, when the peer holds that version.
    pub async fn locate(&self, iid: IllustId, url: &str) -> Result<Option<PathBuf>> {
        let data: Option<String> = sqlx::query_scalar("SELECT data FROM Illust WHERE id = ?")
            .bind(iid)
            .fetch_optional(&self.pool)
            .await?;
        let Some(data) = data else {
            return Ok(None);
        };
        if serde_json::from_str::<Pages>(&data)?.p0().as_deref() != Some(url) {
            return Ok(None);
        }
        let (_, file) = url.rsplit_once('/').unwrap_or(("", url));
        let path = self.pix_dir.join(file);
        Ok(fs::try_exists(&path).await?.then_some(path))
    }
}

#[cfg(test)]
mod tests {
    use super::Peer;
    use crate::config::PeerConfig;
    use serde_json::json;
    use sqlx::SqlitePool;
    use sqlx::sqlite::SqliteConnectOptions;
    use std::path::PathBuf;

    const URL: &str = "https://i.pximg.net/img-original/img/2024/01/02/03/04/05/12345678_p0.png";
    const REUPLOAD: &str =
        "https://i.pximg.net/img-original/img/2024/01/02/03/04/06/12345678_p0.png";
    const P1: &str = "https://i.pximg.net/img-original/img/2024/01/02/03/04/05/12345678_p1.png";

    async fn peer(name: &str, illust: &serde_json::Value, file: Option<&str>) -> Peer {
        let dir = std::env::temp_dir().join(format!("pvg-peer-{}-{name}", std::process::id()));
        _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir(&dir).unwrap();
        let db = dir.join("state.db");
        let pool = SqlitePool::connect_with(
            SqliteConnectOptions::new()
                .filename(&db)
                .create_if_missing(true),
        )
        .await
        .unwrap();
        sqlx::query("CREATE TABLE Illust (id INTEGER PRIMARY KEY, data TEXT NOT NULL)")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO Illust VALUES (12345678, ?)")
            .bind(illust.to_string())
            .execute(&pool)
            .await
            .unwrap();
        pool.close().await;
        if let Some(file) = file {
            std::fs::write(dir.join(file), b"png").unwrap();
        }
        Peer::open(PeerConfig { db, pix_dir: dir }).await.unwrap()
    }

    fn single(url: &str) -> serde_json::Value {
        json!({"meta_single_page": {"original_image_url": url}, "meta_pages": []})
    }

    fn multi(urls: &[&str]) -> serde_json::Value {
        json!({"meta_single_page": {}, "meta_pages": urls.iter()
            .map(|u| json!({"image_urls": {"original": u}})).collect::<Vec<_>>()})
    }

    fn file(path: Option<PathBuf>) -> Option<String> {
        path.map(|p| p.file_name().unwrap().to_str().unwrap().to_owned())
    }

    #[tokio::test]
    async fn takes_the_version_the_peer_holds() {
        let p = peer("single", &single(URL), Some("12345678_p0.png")).await;
        assert_eq!(
            file(p.locate(12_345_678, URL).await.unwrap()).as_deref(),
            Some("12345678_p0.png")
        );
        let p = peer("multi", &multi(&[URL, P1]), Some("12345678_p0.png")).await;
        assert!(p.locate(12_345_678, URL).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn skips_a_re_upload_under_the_same_basename() {
        let p = peer("reupload", &single(URL), Some("12345678_p0.png")).await;
        assert_eq!(p.locate(12_345_678, REUPLOAD).await.unwrap(), None);
    }

    #[tokio::test]
    async fn skips_later_pages_unknown_illusts_and_missing_files() {
        let p = peer("pages", &multi(&[URL, P1]), Some("12345678_p1.png")).await;
        assert_eq!(p.locate(12_345_678, P1).await.unwrap(), None);
        assert_eq!(p.locate(1, URL).await.unwrap(), None);
        assert_eq!(p.locate(12_345_678, URL).await.unwrap(), None);
    }
}
