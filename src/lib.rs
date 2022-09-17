mod pixiv;

#[cfg(test)]
mod tests {
    use crate::pixiv::aapi::{UserBookmarksIllust, UserBookmarksIllustResponse};
    use crate::pixiv::client;
    use crate::pixiv::model::from_response;
    use log::info;

    #[tokio::test]
    async fn test_main() {
        if std::env::var("RUST_LOG").is_err() {
            std::env::set_var("RUST_LOG", "info");
        }
        pretty_env_logger::init_timed();
        let token = "dxRQMY-J3I8Ty9VQd_SoMLt_U0tm8RTjna7O_UOr03w";
        let api = client::AuthedClient::new(token).await.unwrap();
        info!("got {:?}", api.state);
        let uid = 13889701.to_string();
        let req = UserBookmarksIllust::builder().user_id(&*uid).build();
        info!("got {:?}", req);
        let res = api.user_bookmarks_illust(req).await.unwrap();
        info!("got {:?}", res);
        let res: UserBookmarksIllustResponse = from_response(res).unwrap();
        info!("got {:?}", res);
    }
}
