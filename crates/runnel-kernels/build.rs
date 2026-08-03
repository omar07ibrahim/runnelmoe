use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

const SOURCE: &str = "native/bf16_gemv.c";
const HEADER: &str = "include/runnel_kernels.h";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={SOURCE}");
    println!("cargo:rerun-if-changed={HEADER}");
    println!("cargo:rustc-check-cfg=cfg(runnel_native_avx2)");

    if !eligible_native_build() {
        return;
    }

    let out_dir = required_path("OUT_DIR");
    let object = out_dir.join("bf16_gemv.o");
    let archive = out_dir.join("librunnel_kernels.a");
    run(
        Command::new("cc")
            .arg("-std=c11")
            .arg("-O3")
            .arg("-fPIC")
            .arg("-ffp-contract=off")
            .arg("-fno-fast-math")
            .arg("-Wall")
            .arg("-Wextra")
            .arg("-Wpedantic")
            .arg("-Werror")
            .arg("-Wconversion")
            .arg("-Wshadow")
            .arg("-Wstrict-prototypes")
            .arg("-Wmissing-prototypes")
            .arg("-Wformat=2")
            .arg("-Wundef")
            .arg("-Wwrite-strings")
            .arg("-Iinclude")
            .arg("-c")
            .arg(SOURCE)
            .arg("-o")
            .arg(&object),
        "C compiler",
    );
    run(
        Command::new("ar").arg("crs").arg(&archive).arg(&object),
        "archiver",
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=runnel_kernels");
    println!("cargo:rustc-cfg=runnel_native_avx2");
}

fn eligible_native_build() -> bool {
    if env::var_os("CARGO_FEATURE_NATIVE_AVX2").is_none() {
        return false;
    }

    let host = required("HOST");
    let target = required("TARGET");
    host == target
        && required("CARGO_CFG_TARGET_ARCH") == "x86_64"
        && required("CARGO_CFG_TARGET_OS") == "linux"
        && required("CARGO_CFG_TARGET_ENV") == "gnu"
}

fn required(name: &str) -> String {
    env::var(name)
        .unwrap_or_else(|_| panic!("Cargo did not provide required environment variable {name}"))
}

fn required_path(name: &str) -> PathBuf {
    PathBuf::from(required(name))
}

fn run(command: &mut Command, description: &str) {
    let program = command.get_program().to_owned();
    let status = command.status().unwrap_or_else(|error| {
        panic!(
            "failed to invoke required {description} {}: {error}",
            Path::new(&program).display()
        )
    });
    require_success(status, description);
}

fn require_success(status: ExitStatus, description: &str) {
    assert!(
        status.success(),
        "required {description} failed with {status}"
    );
}
