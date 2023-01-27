use reqwest::{Client, Method, RequestBuilder, Url};

struct Version {
    prefix: String,
}

type Result<T> = std::result::Result<T, url::ParseError>;
pub(crate) type SimpleEndpoint = (Method, Url);

impl Version {
    fn new<T: Into<String>>(prefix: T) -> Self {
        Self {
            prefix: prefix.into(),
        }
    }

    fn req(&self, method: Method, path: &str) -> Result<SimpleEndpoint> {
        let url = format!("{}/{}", self.prefix, path);
        Ok((method, Url::parse(&url)?))
    }

    fn get(&self, path: &str) -> Result<SimpleEndpoint> {
        self.req(Method::GET, path)
    }

    fn post(&self, path: &str) -> Result<SimpleEndpoint> {
        self.req(Method::POST, path)
    }
}

pub trait Endpoint {
    fn request(&self, client: &Client) -> RequestBuilder;
}

impl Endpoint for SimpleEndpoint {
    fn request(&self, client: &Client) -> RequestBuilder {
        client.request(self.0.clone(), self.1.clone())
    }
}

#[derive(Debug, Clone)]
pub struct ApiEndpoint {
    pub auth: SimpleEndpoint,
    pub user_bookmarks_illust: SimpleEndpoint,
}

impl ApiEndpoint {
    pub fn with_hosts(app_host: Option<&str>, oauth_host: Option<&str>) -> Result<Self> {
        let app_host = app_host.unwrap_or("https://app-api.pixiv.net");
        let oauth_host = oauth_host.unwrap_or("https://oauth.secure.pixiv.net");
        let appv1 = Version::new(format!("{app_host}/v1"));
        // let appv2 = Version::new(format!("{}/v2", app_host));
        let oauth = Version::new(oauth_host);
        Ok(Self {
            auth: oauth.post("auth/token")?,
            user_bookmarks_illust: appv1.get("user/bookmarks/illust")?,
        })
    }

    pub fn new() -> Self {
        Self::with_hosts(None, None).unwrap()
    }
}
