use crate::endpoint::{ApiEndpoint, Endpoint};
use crate::error::Result;
use crate::model::{Response, User, from_response};
use crate::oauth::{AuthSuccess, auth};
use log::debug;
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::{Client as Http, RequestBuilder};
use serde::{Deserialize, Serialize};
use serde_with::TimestampMilliSeconds;
use serde_with::serde_as;
use std::cmp::max;
use std::fmt::Debug;
use std::time::SystemTime;
use tokio::time::Duration;

#[serde_as]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AuthedState {
    pub access_header: String,
    pub refresh_token: String,
    #[serde_as(as = "TimestampMilliSeconds<i64>")]
    pub expires_at: SystemTime,
    pub user: User,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct GuestState {}

pub trait ApiState {
    fn handle(&self, req: RequestBuilder) -> RequestBuilder;
}

impl ApiState for AuthedState {
    fn handle(&self, req: RequestBuilder) -> RequestBuilder {
        req.header("Authorization", &self.access_header)
    }
}

impl ApiState for GuestState {
    fn handle(&self, req: RequestBuilder) -> RequestBuilder {
        req
    }
}

#[derive(Debug, Clone)]
pub struct AuthResult {
    resp: Response,
    time: SystemTime,
}

impl AuthedState {
    pub fn new(res: AuthResult) -> Result<Self> {
        let resp: AuthSuccess = from_response(res.resp)?;
        let resp = resp.response;
        debug!(
            "authed: {}, {} ({}) in {}",
            resp.user.id, resp.user.name, resp.user.account, resp.expires_in,
        );
        Ok(Self {
            access_header: format!("Bearer {}", resp.access_token),
            refresh_token: resp.refresh_token,
            expires_at: res.time + Duration::from_secs(u64::from(max(0, resp.expires_in - 30))),
            user: resp.user,
        })
    }

    #[must_use]
    pub fn expired(&self) -> bool {
        SystemTime::now() > self.expires_at
    }
}

#[derive(Debug, Clone)]
pub struct Client<S: ApiState> {
    http: Http,
    pub(crate) api: ApiEndpoint,
    pub state: S,
}

pub type AuthedClient = Client<AuthedState>;
pub type GuestClient = Client<GuestState>;

impl<S: ApiState> Client<S> {
    fn make(state: S) -> Self {
        let mut headers = HeaderMap::with_capacity(3);
        headers.insert("app-os", HeaderValue::from_static("ios"));
        headers.insert("app-os-version", HeaderValue::from_static("14.6"));

        Self {
            http: Http::builder()
                .default_headers(headers)
                .user_agent("PixivIOSApp/7.13.3 (iOS 14.6; iPhone13,2)")
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
            api: ApiEndpoint::new(),
            state,
        }
    }

    fn guest_call(&self, endpoint: &impl Endpoint) -> RequestBuilder {
        endpoint.request(&self.http)
    }

    pub(crate) fn call(&self, endpoint: &impl Endpoint) -> RequestBuilder {
        self.state.handle(self.guest_call(endpoint))
    }

    async fn do_auth(&self, refresh_token: &str) -> Result<Response> {
        let req = self.guest_call(&self.api.auth);
        auth(req, refresh_token).await
    }

    pub async fn raw_auth(&self, refresh_token: &str) -> Result<AuthResult> {
        let now = SystemTime::now();
        let resp = self.do_auth(refresh_token).await?;
        debug!("auth: {resp:?}");
        Ok(AuthResult { resp, time: now })
    }

    async fn auth(&self, refresh_token: &str) -> Result<AuthedState> {
        let s = AuthedState::new(self.raw_auth(refresh_token).await?)?;
        Ok(s)
    }
}

impl Default for GuestClient {
    fn default() -> Self {
        Self::make(GuestState::default())
    }
}

impl GuestClient {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn into_authed(self, token: AuthedState) -> AuthedClient {
        Client {
            http: self.http,
            api: self.api,
            state: token,
        }
    }
}

impl AuthedClient {
    pub async fn new(refresh_token: &str) -> Result<Self> {
        let r = GuestClient::new();
        let s = r.auth(refresh_token).await?;
        Ok(r.into_authed(s))
    }

    #[must_use]
    pub fn load(state: AuthedState) -> Self {
        Self::make(state)
    }

    pub async fn ensure_authed(&mut self) -> Result<()> {
        if self.state.expired() {
            self.state = self.auth(&self.state.refresh_token).await?;
        }
        Ok(())
    }
}
