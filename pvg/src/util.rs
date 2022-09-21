#[macro_export]
macro_rules! critical {
    ($f:expr, $($x:expr),+) => {
        error!(concat!("CRITICAL: ", $f), $($x),+)
    };
}

#[macro_export]
macro_rules! bug {
    ($f:expr, $($x:expr),+) => {
        error!(concat!("BUG: ", $f), $($x),+)
    };
}
