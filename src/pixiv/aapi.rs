use crate::pixiv::endpoint::ApiEndpoint;
use crate::pixiv::model::Response;
use crate::pixiv::session::Session;
use crate::pixiv::Result;

pub struct AppApi {
    session: Session,
    api: ApiEndpoint,
}

impl AppApi {
    pub fn new() -> Self {
        Self {
            session: Session::new(),
            api: ApiEndpoint::new(),
        }
    }

    pub async fn login(&mut self, refresh_token: &str) -> Result<Response> {
        self.session.auth(&self.api.auth, refresh_token).await
    }
}
