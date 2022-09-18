use crate::endpoint::{ApiEndpoint, Endpoint};
use crate::error::Result;
use crate::model::{from_response, Response, User};
use crate::oauth::{auth, AuthSuccess};
use log::info;
use reqwest::{Client as Http, RequestBuilder};
use std::cmp::max;
use std::fmt::Debug;
use tokio::time::{Duration, Instant};

#[derive(Debug)]
pub struct AuthedState {
    pub access_header: String,
    pub refresh_token: String,
    pub expires_at: Instant,
    pub user: User,
}

#[derive(Debug)]
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

#[derive(Debug)]
pub struct AuthResult {
    resp: Response,
    time: Instant,
}

impl AuthedState {
    pub fn new(res: AuthResult) -> Result<Self> {
        let resp: AuthSuccess = from_response(res.resp)?;
        let resp = resp.response;
        Ok(Self {
            access_header: format!("Bearer {}", resp.access_token),
            refresh_token: resp.refresh_token,
            expires_at: res.time + Duration::from_secs(max(0, resp.expires_in - 30) as u64),
            user: resp.user,
        })
    }

    pub fn expired(&self) -> bool {
        Instant::now() > self.expires_at
    }
}

#[derive(Debug)]
pub struct Client<S: ApiState> {
    http: Http,
    pub(crate) api: ApiEndpoint,
    pub state: S,
}

pub type AuthedClient = Client<AuthedState>;
pub type GuestClient = Client<GuestState>;

impl<S: ApiState> Client<S> {
    fn make(state: S) -> Self {
        Self {
            http: Http::new(),
            api: ApiEndpoint::new(),
            state,
        }
    }

    fn guest_call(&self, endpoint: &impl Endpoint) -> RequestBuilder {
        endpoint
            .request(&self.http)
            .header("app-os", "ios")
            .header("app-os-version", "14.6")
            .header("user-agent", "PixivIOSApp/7.13.3 (iOS 14.6; iPhone13,2)")
    }

    pub(crate) fn call(&self, endpoint: &impl Endpoint) -> RequestBuilder {
        self.state.handle(self.guest_call(endpoint))
    }

    async fn do_auth(&self, refresh_token: &str) -> Result<Response> {
        let req = self.guest_call(&self.api.auth);
        auth(req, refresh_token).await
    }

    pub async fn raw_auth(&self, refresh_token: &str) -> Result<AuthResult> {
        let now = Instant::now();
        let resp = self.do_auth(refresh_token).await?;
        info!("auth: {:?}", resp);
        Ok(AuthResult { resp, time: now })
    }

    async fn auth(&self, refresh_token: &str) -> Result<AuthedState> {
        let s = AuthedState::new(self.raw_auth(refresh_token).await?)?;
        Ok(s)
    }
}

impl GuestClient {
    pub fn new() -> Self {
        Self::make(GuestState {})
    }

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

    pub async fn ensure_authed(&mut self) -> Result<()> {
        if self.state.expired() {
            self.state = self.auth(&self.state.refresh_token).await?;
        }
        Ok(())
    }
}
