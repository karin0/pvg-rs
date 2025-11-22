use std::error::Error;
use vergen::{BuildBuilder, CargoBuilder, Emitter, RustcBuilder, SysinfoBuilder};
use vergen_gitcl::GitclBuilder;

fn main() -> Result<(), Box<dyn Error>> {
    let gitcl = GitclBuilder::default().describe(true, true, None).build()?;
    let build = BuildBuilder::default().build_timestamp(true).build()?;
    let cargo = CargoBuilder::default()
        .features(true)
        .opt_level(true)
        .debug(true)
        .build()?;
    let rustc = RustcBuilder::default()
        .semver(true)
        .host_triple(true)
        .build()?;
    let si = SysinfoBuilder::default().os_version(true).build()?;

    Emitter::default()
        .add_instructions(&gitcl)?
        .add_instructions(&build)?
        .add_instructions(&cargo)?
        .add_instructions(&rustc)?
        .add_instructions(&si)?
        .emit()?;

    Ok(())
}
