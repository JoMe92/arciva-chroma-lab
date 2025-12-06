# Changelog

## [0.2.0](https://github.com/JoMe92/chroma-lab/compare/v0.1.0...v0.2.0) (2025-12-06)


### Features

* prepare repository for public release ([c043d56](https://github.com/JoMe92/chroma-lab/commit/c043d5634c0b225dd00f7fb68361877386528b50))

## [0.2.2](https://github.com/JoMe92/chroma-lab/compare/v0.2.1...v0.2.2) (2025-12-06)


### Bug Fixes

* force release to apply package config ([7b54aca](https://github.com/JoMe92/chroma-lab/commit/7b54acaa86a403a1d287fd1dfc8b8422327e6712))

## [0.2.1](https://github.com/JoMe92/chroma-lab/compare/v0.2.0...v0.2.1) (2025-12-06)


### Bug Fixes

* force release for public launch ([9147050](https://github.com/JoMe92/chroma-lab/commit/91470506f2cdbc97c0044d50cf36b96ac20fd5c9))

## [0.2.0](https://github.com/JoMe92/chroma-lab/compare/v0.1.0...v0.2.0) (2025-12-06)


### Features

* add repository URL to quickfix-renderer and enable manual publish workflow dispatch ([9515186](https://github.com/JoMe92/chroma-lab/commit/9515186430e37de0d534881ba32db9c600de9ecd))

## 0.1.0 (2025-12-06)


### Features

* 5 wasm bindings and typescript friendly api ([7b07588](https://github.com/JoMe92/chroma-lab/commit/7b075888ace2d969a2bb53e4a05fea804fbdd3da))
* Add image processing example and update `process_frame` to return processed image data with dimensions. ([92645f8](https://github.com/JoMe92/chroma-lab/commit/92645f827a9d5a83448eeedc6836e20e26c7b0af))
* Add MIT license and metadata to `quickfix-renderer` and remove its alias and `topLevelAwait` plugin from `vite.config.ts`.Refs; [#5](https://github.com/JoMe92/chroma-lab/issues/5) ([69e4189](https://github.com/JoMe92/chroma-lab/commit/69e41895061243061ba4a5ff050a711c37043ff3))
* Add quickfix-renderer package and install Node.js dependencies for the api-test example. ([81e6ee6](https://github.com/JoMe92/chroma-lab/commit/81e6ee69bdd96aed1b9dc95e07eb49a730369e62)), closes [#5](https://github.com/JoMe92/chroma-lab/issues/5)
* Add Rust unit tests and React component tests with Vitest, including necessary configuration and dependencies. ([857dde4](https://github.com/JoMe92/chroma-lab/commit/857dde421e78161e0fcdc031500e6dc11f6f3425)), closes [#4](https://github.com/JoMe92/chroma-lab/issues/4)
* Explicitly resize canvas in TypeScript and Rust, and re-initialize WebGL context when the target canvas element changes. ([9963261](https://github.com/JoMe92/chroma-lab/commit/9963261cd60ab206f313f12e944b8b8e3314214f)), closes [#4](https://github.com/JoMe92/chroma-lab/issues/4)
* gpu render pipeline with webgpu and webgl2 fallback ([9a6ddc7](https://github.com/JoMe92/chroma-lab/commit/9a6ddc7230bd7bacbb005042f93535739de69a61))
* Implement and document web worker architecture for quickfix renderer, refining client-side interaction. ([ec3aa51](https://github.com/JoMe92/chroma-lab/commit/ec3aa51ccc3e70f2e90c005b93f705b9371f7d74)), closes [#6](https://github.com/JoMe92/chroma-lab/issues/6)
* Implement multi-backend rendering with WebGPU, WebGL2, and CPU fallbacks via a new `Renderer` trait. ([2d7c82b](https://github.com/JoMe92/chroma-lab/commit/2d7c82b0b6947b1298e9ba5a3557b9af2a107708)), closes [#4](https://github.com/JoMe92/chroma-lab/issues/4)
* Implement stateful image rendering in worker to reduce data transfer overhead by setting image once. ([b3680ea](https://github.com/JoMe92/chroma-lab/commit/b3680ea5ead339a32616053460d09b102522a700)), closes [#6](https://github.com/JoMe92/chroma-lab/issues/6)
* Offload QuickFixRenderer WASM operations to a Web Worker for improved main thread performance. ([04c14ec](https://github.com/JoMe92/chroma-lab/commit/04c14ec423d263d7e5492706766858b297b313e8)), closes [#4](https://github.com/JoMe92/chroma-lab/issues/4)
* **quickfix-renderer:** implement CPU-based adjustment pipeline ([c8d84a6](https://github.com/JoMe92/chroma-lab/commit/c8d84a6e6d6f2bdf5b8d4a845f4be2bd0f433043))
* **quickfix-renderer:** implement CPU-based adjustment pipeline ([9f6abf3](https://github.com/JoMe92/chroma-lab/commit/9f6abf306b358afe1f9f671f6960b4e520ecbd99)), closes [#3](https://github.com/JoMe92/chroma-lab/issues/3)
* **renderer:** implement Web Worker wrapper and TypeScript client ([c3bf0ca](https://github.com/JoMe92/chroma-lab/commit/c3bf0ca65535d7b4d0e64bfe605057177ff1e22d)), closes [#6](https://github.com/JoMe92/chroma-lab/issues/6)
* **repo:** bootstrap Rust/WASM renderer with example app and CI ([f342c16](https://github.com/JoMe92/chroma-lab/commit/f342c167726166b06cd00e3de216f2a978eefe77))
* **repo:** bootstrap Rust/WASM renderer with example app and CI ([e67264b](https://github.com/JoMe92/chroma-lab/commit/e67264b5485e35f7e6e288e3e27b10f3f99800ff)), closes [#2](https://github.com/JoMe92/chroma-lab/issues/2)
* Update quickfix-renderer API, add usage documentation, and remove old API test example. ([fec605b](https://github.com/JoMe92/chroma-lab/commit/fec605bd912612c38eb580d3e3f69560870852af)), closes [#5](https://github.com/JoMe92/chroma-lab/issues/5)
