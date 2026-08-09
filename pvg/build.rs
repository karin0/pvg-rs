use std::error::Error;
use vergen::{Build, Cargo, Emitter, Rustc, Sysinfo};
use vergen_gitcl::Gitcl;

fn main() -> Result<(), Box<dyn Error>> {
    let gitcl = Gitcl::builder().describe(true, true, None).build();
    let build = Build::builder().build_timestamp(true).build();
    let cargo = Cargo::builder()
        .features(true)
        .opt_level(true)
        .debug(true)
        .build();
    let rustc = Rustc::builder().semver(true).host_triple(true).build();
    let si = Sysinfo::builder().os_version(true).build();

    Emitter::default()
        .add_instructions(&gitcl)?
        .add_instructions(&build)?
        .add_instructions(&cargo)?
        .add_instructions(&rustc)?
        .add_instructions(&si)?
        .emit()?;

    Ok(())
}
