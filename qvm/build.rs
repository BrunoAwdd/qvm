use std::path::PathBuf;
use std::process::Command;
use std::fs;

fn main() {
    println!("cargo:warning=🚧 Building CUDA kernels...");
    let base_dir = PathBuf::from("src/gates");

    // Recursively search for .cu files
    let pattern = format!("{}/**/*.cu", base_dir.display());
    for entry in glob::glob(&pattern).expect("Failed to read glob pattern") {
        let cu_path = entry.expect("Invalid path");
        let ptx_path = cu_path.with_extension("ptx");

        // Ensure cargo only reruns this script when .cu changes
        println!("cargo:rerun-if-changed={}", cu_path.display());

        // Skip recompilation if .ptx is newer
        let should_compile = match (fs::metadata(&cu_path), fs::metadata(&ptx_path)) {
            (Ok(cu_meta), Ok(ptx_meta)) => {
                let cu_time = cu_meta.modified().unwrap();
                let ptx_time = ptx_meta.modified().unwrap();
                cu_time > ptx_time
            }
            _ => true, // .ptx doesn't exist
        };

        if should_compile {
            println!(
                "cargo:warning=Compiling CUDA → PTX: {} → {}",
                cu_path.display(),
                ptx_path.display()
            );

            let status = Command::new("nvcc")
                .args(["-ptx"])
                .arg(&cu_path)
                .arg("-o")
                .arg(&ptx_path)
                .status()
                .expect("Failed to execute nvcc");

            if !status.success() {
                panic!("nvcc failed on {}", cu_path.display());
            }
        }
    }
}
