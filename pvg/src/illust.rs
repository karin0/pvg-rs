use itertools::Itertools;
use pixiv::IllustId;
use pixiv::model as api;
use std::collections::BTreeMap;
use std::collections::HashMap;
use string_interner::{DefaultSymbol, StringInterner, backend::BucketBackend as Backend};

type TagId = DefaultSymbol;
type UserNameId = DefaultSymbol;

struct User {
    cnt: BTreeMap<UserNameId, u32>,
    max: (UserNameId, u32),
}

impl User {
    fn new(name: UserNameId) -> Self {
        let mut cnt = BTreeMap::new();
        cnt.insert(name, 1);
        User {
            cnt,
            max: (name, 1),
        }
    }

    fn push(&mut self, name: UserNameId) {
        let c = self.cnt.entry(name).or_insert(0);
        *c += 1;
        if *c > self.max.1 {
            self.max = (name, *c);
        }
    }
}

#[derive(Debug, Clone)]
pub struct IllustData {
    pub id: IllustId,
    pub title: String,
    pub user_id: u32,
    pub original_user_name: UserNameId,
    pub width: u16,
    pub height: u16,
    pub page_count: u16,
    pub tags: Vec<TagId>,
    pub sanity_level: u8,
    pub x_restrict: u8,
    pub visible: bool,
    pub create_date: String,
}

pub struct IllustService {
    users: HashMap<u32, User>,
    tags: StringInterner<Backend>,
    user_names: StringInterner<Backend>,
}

impl IllustService {
    pub fn new() -> Self {
        Self {
            users: HashMap::new(),
            tags: StringInterner::<Backend>::new(),
            user_names: StringInterner::<Backend>::new(),
        }
    }

    pub fn resolve(&mut self, data: api::Illust, new: bool) -> IllustData {
        let user_id = data.user.id;
        let user_name = self.user_names.get_or_intern(data.user.name.trim());

        self.users
            .entry(user_id)
            .and_modify(|u| {
                if new {
                    u.push(user_name);
                }
            })
            .or_insert_with(|| User::new(user_name));

        let mut tags = data
            .tags
            .into_iter()
            .map(|t| self.tags.get_or_intern(t.name))
            .collect_vec();
        tags.shrink_to_fit();

        let mut r = IllustData {
            id: data.id,
            title: data.title,
            user_id,
            original_user_name: user_name,
            width: data.width as u16,
            height: data.height as u16,
            page_count: data.page_count as u16,
            tags,
            sanity_level: data.sanity_level as u8,
            x_restrict: data.x_restrict as u8,
            visible: data.visible,
            create_date: data.create_date,
        };
        r.title.shrink_to_fit();
        r.create_date.shrink_to_fit();
        r
    }

    pub fn get_user_name(&self, data: &IllustData) -> &str {
        let user = &self.users[&data.user_id];
        let id = user.max.0;
        self.user_names.resolve(id).unwrap()
    }

    pub fn get_tags<'a>(&'a self, data: &'a IllustData) -> Box<dyn Iterator<Item = &'a str> + 'a> {
        let id = self.users[&data.user_id].max.0;
        let it = data.tags.iter().map(|&id| self.tags.resolve(id).unwrap());
        if id == data.original_user_name {
            Box::new(it)
        } else {
            Box::new(
                std::iter::once(self.user_names.resolve(data.original_user_name).unwrap())
                    .chain(it),
            )
        }
    }

    pub fn get_original_user_name(&self, data: &IllustData) -> &str {
        self.user_names.resolve(data.original_user_name).unwrap()
    }
}
