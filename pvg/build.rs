use std::error::Error;
use vergen_gitcl::{Emitter, GitclBuilder};

fn main() -> Result<(), Box<dyn Error>> {
    let gitcl = GitclBuilder::default().describe(true, true, None).build()?;
    Emitter::default().add_instructions(&gitcl)?.emit()?;

    let features: Vec<String> = std::env::vars()
        .filter_map(|(key, _)| key.strip_prefix("CARGO_FEATURE_").map(str::to_lowercase))
        .collect();
    println!("cargo:rustc-env=PVG_FEATURES={}", features.join(","));

    Ok(())
}
