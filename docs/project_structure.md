# Project Structure Guide

This guide is designed for developers who are familiar with general software development (JavaScript/TypeScript, C++, etc.) but are new to **Rust** and **WebAssembly**.

## High-Level Overview

This repository is a **hybrid workspace** combining a Rust crate (library) and a TypeScript/React example application.

```text
.
├── quickfix-renderer/       # 🦀 The Rust Source Code (The "Backend" logic)
│   ├── Cargo.toml           # Dependency manager (like package.json)
│   ├── src/                 # Rust source files
│   └── pkg/                 # 📦 Generated WASM package (Output)
├── examples/
│   └── minimal-viewer/      # ⚛️ React App that uses the WASM package
├── docs/                    # Documentation
├── pixi.toml                # Developer environment configuration
└── package.json             # Workspace-level scripts
```

## The Rust Crate (`quickfix-renderer/`)

This is where the core logic lives. It compiles into a WebAssembly module that can be imported into JavaScript.

### `Cargo.toml`

This is the **manifest file** for Rust, similar to `package.json` in Node.js.

* **`[package]`**: Metadata (name, version).
* **`[lib]`**: Defines this project as a library (`cdylib` is required for WASM).
* **`[dependencies]`**: Lists external Rust crates (libraries).
  * `wasm-bindgen`: The magic tool that generates the JavaScript <-> Rust bridge.
  * `wgpu`: Cross-platform GPU API (WebGPU/Vulkan/Metal/DX12).
  * `glow`: WebGL2 bindings.
  * `image`: Image processing library (used for CPU fallback).

### `src/` Directory

* **`lib.rs`**: The **entry point** of the library.
  * It defines the public API exposed to JavaScript.
  * Look for `#[wasm_bindgen]` attributes – these mark functions and classes that will be available in JS.
  * It handles the "Orchestration": deciding whether to use WebGPU, WebGL2, or CPU.
* **`operations.rs`**: The **CPU implementation**.
  * Pure Rust code that manipulates pixel buffers directly.
  * Serves as the "Reference Implementation" for correctness.
* **`webgpu.rs`**: The **WebGPU backend**.
  * Manages the GPU device, pipelines, and buffers using the `wgpu` crate.
* **`webgl.rs`**: The **WebGL2 backend**.
  * Manages the GL context and shaders using the `glow` crate.
* **`shaders.rs`**: Contains the shader source code (WGSL for WebGPU, GLSL for WebGL2) as string constants.

### `pkg/` Directory (Generated)

This folder is **not** in source control (usually). It is created when you run `wasm-pack build`.
It contains:

* `quickfix_renderer_bg.wasm`: The binary WebAssembly code.
* `quickfix_renderer.js`: The JavaScript "glue code" that loads the WASM and provides nice JS classes/functions to call it.
* `package.json`: Makes this folder a valid NPM package.

## The Example App (`examples/minimal-viewer/`)

This is a standard **Vite + React + TypeScript** application.

* **`package.json`**:

    ```json
    "dependencies": {
      "quickfix-renderer": "file:../../quickfix-renderer/pkg"
    }
    ```

    This line is key. It tells the package manager to install the `quickfix-renderer` package directly from the local `pkg` folder, rather than from the npm registry.

* **`vite.config.ts`**:
  * Includes `vite-plugin-wasm` and `vite-plugin-top-level-await` to handle correct loading of WASM modules in the browser.

## The Build Flow

1. **Rust Compilation**:
    You run `pixi run build:wasm` (wraps `wasm-pack build`).
    * `rustc` compiles Rust code -> WASM binary.
    * `wasm-bindgen` generates the JS wrapper.
    * Output is placed in `quickfix-renderer/pkg`.

2. **Frontend Bundling**:
    The example app imports `quickfix-renderer`.
    * Vite sees the import.
    * It resolves it to the local `pkg` folder.
    * It bundles the JS glue code and ensures the `.wasm` file is served correctly.

## Key Rust Concepts for JS Developers

* **Ownership & Borrowing**: You might see `&self` or `&mut self`. This is Rust's way of managing memory safety without a Garbage Collector.
  * `&self`: Read-only access.
  * `&mut self`: Mutable (write) access.
* **`Result<T, E>`**: Rust doesn't use Exceptions for standard errors. It returns a `Result` type. You'll see `?` used often (e.g., `function_call()?`), which means "if this fails, return the error immediately, otherwise give me the value".
* **`Option<T>`**: No `null` or `undefined`. A value is either `Some(value)` or `None`.
* **`pub`**: Functions are private by default. `pub` makes them public (visible to other modules). `#[wasm_bindgen]` makes them public to JavaScript.
