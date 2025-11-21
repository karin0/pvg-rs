use std::sync::atomic::AtomicU32;

pub type Key = pixiv::IllustId;
pub type Pos = u32;
pub type Idx = u32;
pub type Item<'a> = (Key, &'a str);

pub static STATS: AtomicU32 = AtomicU32::new(0);
