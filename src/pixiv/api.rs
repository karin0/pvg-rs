use reqwest::{Client, Method, Request, Url};

struct Version {
    prefix: String,
}

impl Version {
    fn new<T: Into<String>>(prefix: T) -> Self {
        Self {
            prefix: prefix.into(),
        }
    }

    fn req(&self, method: Method, path: &str) -> Result<Request, url::ParseError> {
        let url = format!("{}/{}", self.prefix, path);
        let resp = Request::new(method, Url::parse(&url)?);
        Ok(resp)
    }

    fn get(&self, path: &str) -> Result<Request, url::ParseError> {
        self.req(Method::GET, path)
    }

    fn post(&self, path: &str) -> Result<Request, url::ParseError> {
        self.req(Method::POST, path)
    }
}

struct API {
    appv1_host: String,
    appv2_host: String,
    auth_host: String,
    client: Client,
}

impl API {
    fn new(
        self,
        app_host: Option<&str>,
        public_host: Option<&str>,
        oauth_host: Option<&str>,
    ) -> Self {
        let app_host = app_host.unwrap_or("https://app-api.pixiv.net");
        let public_host = public_host.unwrap_or("https://public-api.secure.pixiv.net");
        let oauth_host = oauth_host.unwrap_or("https://oauth.secure.pixiv.net");
        Self {
            appv1_host: format!("{}/v1", app_host),
            appv2_host: format!("{}/v2", app_host),
            auth_host: format!("{}/auth/token", oauth_host),
            client: Client::new(),
        }
    }

    fn login(&mut self, username: &str, refresh_token: &str) {}
}
