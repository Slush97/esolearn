fn main() {
    // When the `dnnl` feature is enabled, link against Intel oneDNN.
    #[cfg(feature = "dnnl")]
    {
        println!("cargo:rustc-link-lib=dylib=dnnl");
    }

    // MKL takes priority over generic OpenBLAS when both are active.
    #[cfg(feature = "mkl")]
    {
        // Single Dynamic Library mode — simplest MKL linkage.
        // Requires libmkl_rt.so on LD_LIBRARY_PATH or in system lib dirs.
        println!("cargo:rustc-link-lib=dylib=mkl_rt");
    }

    // Generic BLAS (OpenBLAS) — only if mkl is not active.
    #[cfg(all(feature = "blas", not(feature = "mkl")))]
    {
        println!("cargo:rustc-link-lib=dylib=openblas");
    }
}
