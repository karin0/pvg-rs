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
