# Rust/WASM Quick Fix Renderer

A standalone Rust/WASM renderer for image adjustments (Geometry, Crop, Rotate, Exposure, Color, Grain).

## Prerequisites

- [Pixi](https://prefix.dev/) (Package Manager)

## Getting Started

1. **Clone the repository:**

    ```bash
    git clone https://github.com/JoMe92/chroma-lab.git
    cd chroma-lab
    ```

2. **Install dependencies:**

    ```bash
    pixi install
    ```

3. **Run the example app:**

    ```bash
    pixi run dev
    ```

    This will start the development server for the `minimal-viewer` example. Open your browser to the provided URL (usually `http://localhost:5173`).

## Development Commands

- `pixi run dev`: Start the example app development server.
- `pixi run test`: Run Rust tests.
- `pixi run lint`: Run `cargo fmt` and `cargo clippy`.
- `pixi run build`: Build the WASM package and bundle it.

## Architecture

- **`quickfix-renderer`**: Rust crate compiled to WASM.
- **`examples/minimal-viewer`**: React/Vite app demonstrating usage via Web Worker.

See [Web Worker Architecture](docs/web-worker-architecture.md) for a deep dive into the worker integration.
See [Renderer Integration Guide](docs/renderer-integration.md) for detailed integration steps and data flow diagrams.

## API Documentation

Automated API documentation can be generated for both the Rust crate and the TypeScript client.

### Rust Docs
```bash
cargo doc --open
```

### TypeScript Docs
```bash
pnpm run docs
```
The documentation will be generated in `docs/ts`.

## Feature Flags

- `cpu` (default): Deterministic CPU-based rendering.
- `webgl2`: WebGL2 backend (future).
- `webgpu`: WebGPU backend (future).

## License

MIT
