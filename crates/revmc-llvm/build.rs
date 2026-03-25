#![allow(missing_docs)]

fn main() {
    let cxxflags =
        std::process::Command::new("llvm-config").arg("--cxxflags").output().expect("llvm-config");
    let cxxflags = String::from_utf8(cxxflags.stdout).unwrap();

    cc::Build::new()
        .cpp(true)
        .flags(cxxflags.split_whitespace())
        .file("cpp/lib.cpp")
        .compile("revmc_llvm_cpp");
}
