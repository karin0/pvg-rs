#[macro_export]
macro_rules! critical {
    ($f:tt $(,$x:expr)*) => {
        error!(concat!("CRITICAL: ", $f) $(,$x)*)
    };
}

#[macro_export]
macro_rules! bug {
    ($f:tt $(,$x:expr)*) => {
        error!(concat!("BUG: ", $f) $(,$x)*)
    };
}

pub fn normalized(s: &str) -> String {
    use unicode_normalization::UnicodeNormalization;

    s.to_lowercase()
        .cjk_compat_variants()
        .collect::<String>()
        .nfkc()
        .collect::<String>()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
}
