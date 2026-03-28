// SPDX-License-Identifier: MIT OR Apache-2.0
//! scry-cv workbench — Tauri desktop application.

#![allow(clippy::wildcard_imports)]

mod commands;
mod generators;
mod ops;
mod state;

fn main() {
    tauri::Builder::default()
        .manage(state::AppState {
            pipeline: std::sync::Mutex::new(state::Pipeline::new()),
        })
        .invoke_handler(tauri::generate_handler![
            commands::list_operations,
            commands::set_source,
            commands::add_step,
            commands::update_step,
            commands::remove_step,
            commands::get_step,
            commands::get_pipeline,
        ])
        .run(tauri::generate_context!())
        .expect("error running scry-app");
}
