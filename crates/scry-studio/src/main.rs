// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod state;

use state::AppState;

fn main() {
    tauri::Builder::default()
        .manage(AppState::new())
        .invoke_handler(tauri::generate_handler![
            commands::backend_info,
            commands::llm::llm_load,
            commands::llm::llm_generate,
            commands::llm::llm_tokenize,
            commands::llm::llm_decode,
            commands::llm::llm_logits,
            commands::vision::resnet_load,
            commands::vision::resnet_classify,
            commands::vision::scrfd_load,
            commands::vision::scrfd_detect,
            commands::diffusion::diffusion_load,
            commands::diffusion::diffusion_generate,
        ])
        .run(tauri::generate_context!())
        .expect("error while running scry-studio");
}
