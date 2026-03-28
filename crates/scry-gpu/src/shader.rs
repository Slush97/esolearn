//! WGSL → SPIR-V compilation via naga.

use crate::error::{GpuError, Result};

/// A compiled shader module ready for dispatch.
pub struct CompiledShader {
    /// SPIR-V binary (words, not bytes).
    pub spirv: Vec<u32>,
    /// Parsed naga module (retained for reflection).
    pub module: naga::Module,
    /// Entry point name.
    pub _entry_point: String,
}

/// Compile a WGSL source string to SPIR-V.
///
/// Validates the module and extracts binding metadata.
pub fn compile_wgsl(source: &str, entry_point: &str) -> Result<CompiledShader> {
    // Parse WGSL
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|e| GpuError::ShaderCompilation(format!("{e}")))?;

    // Validate
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| GpuError::ShaderCompilation(format!("{e}")))?;

    // Check entry point exists
    let ep_exists = module.entry_points.iter().any(|ep| ep.name == entry_point);
    if !ep_exists {
        return Err(GpuError::MissingEntryPoint {
            name: entry_point.to_string(),
        });
    }

    // Emit SPIR-V
    let spirv = naga::back::spv::write_vec(
        &module,
        &info,
        &naga::back::spv::Options {
            lang_version: (1, 3),
            ..Default::default()
        },
        None,
    )
    .map_err(|e| GpuError::ShaderCompilation(format!("{e}")))?;

    Ok(CompiledShader {
        spirv,
        module,
        _entry_point: entry_point.to_string(),
    })
}

/// Returns the byte size of the push constant block, or 0 if none.
///
/// Reflects the struct `span` from the naga module's type arena.
pub fn push_constant_size(module: &naga::Module) -> u32 {
    for (_, var) in module.global_variables.iter() {
        if var.space == naga::AddressSpace::PushConstant {
            let ty = &module.types[var.ty];
            if let naga::TypeInner::Struct { span, .. } = ty.inner {
                return span;
            }
        }
    }
    0
}

/// Reflect binding info from a compiled shader.
///
/// Returns the number of storage buffer bindings declared in bind group 0.
pub fn binding_count(module: &naga::Module) -> usize {
    module
        .global_variables
        .iter()
        .filter(|(_, var)| var.binding.is_some())
        .filter(|(_, var)| {
            matches!(
                var.space,
                naga::AddressSpace::Storage { .. } | naga::AddressSpace::Uniform
            )
        })
        .count()
}
