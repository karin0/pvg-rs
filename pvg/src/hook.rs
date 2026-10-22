use serde_json::Value;

#[derive(Debug, Clone)]
pub struct DownloadHook {
    pub urls: Vec<String>,
    pub client: reqwest::Client,
}

impl DownloadHook {
    pub fn new(urls: Vec<String>, client: reqwest::Client) -> Self {
        Self { urls, client }
    }

    pub async fn post_payload(&self, payload: &[Value]) {
        futures::future::join_all(self.urls.iter().map(|url| self.post_one(url, payload))).await;
    }

    async fn post_one(&self, url: &str, payload: &[Value]) {
        match self.client.post(url).json(payload).send().await {
            Ok(resp) => {
                if !resp.status().is_success() {
                    error!(
                        "download hook batch failed: {} {} ({} illusts)",
                        resp.status(),
                        url,
                        payload.len()
                    );
                }
            }
            Err(e) => {
                error!(
                    "download hook batch request failed: {e}: {} ({} illusts)",
                    url,
                    payload.len()
                );
            }
        }
    }
}
