use itertools::Itertools;
use pixiv::IllustId;
use serde_json::Value;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct DownloadHook {
    pub url: String,
    pub client: reqwest::Client,
}

impl DownloadHook {
    pub fn new(url: String, client: reqwest::Client) -> Self {
        Self { url, client }
    }

    pub async fn post_payload(&self, payload: &[Value]) {
        match self.client.post(&self.url).json(payload).send().await {
            Ok(resp) => {
                if !resp.status().is_success() {
                    error!(
                        "download hook batch failed: {} {} ({} illusts)",
                        resp.status(),
                        self.url,
                        payload.len()
                    );
                }
            }
            Err(e) => {
                error!(
                    "download hook batch request failed: {e}: {} ({} illusts)",
                    self.url,
                    payload.len()
                );
            }
        }
    }
}

pub trait DownloadHookState {
    fn on_downloaded(&mut self, _: IllustId) {}

    fn finish(self) -> Option<Vec<IllustId>>
    where
        Self: Sized,
    {
        None
    }
}

pub struct NoDownloadHookState;

impl DownloadHookState for NoDownloadHookState {}

pub struct EnabledDownloadHookState {
    iid_order: Vec<IllustId>,
    downloaded_iids: HashSet<IllustId>,
}

impl EnabledDownloadHookState {
    pub fn new<I>(iids: I) -> Self
    where
        I: IntoIterator<Item = IllustId>,
    {
        let mut iid_seen = HashSet::new();
        let mut iid_order = Vec::new();
        for iid in iids {
            if iid_seen.insert(iid) {
                iid_order.push(iid);
            }
        }
        Self {
            iid_order,
            downloaded_iids: HashSet::new(),
        }
    }
}

impl DownloadHookState for EnabledDownloadHookState {
    fn on_downloaded(&mut self, iid: IllustId) {
        self.downloaded_iids.insert(iid);
    }

    fn finish(self) -> Option<Vec<IllustId>> {
        Some(
            self.iid_order
                .into_iter()
                .filter(|iid| self.downloaded_iids.contains(iid))
                .collect_vec(),
        )
    }
}
