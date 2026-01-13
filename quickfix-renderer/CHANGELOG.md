# Changelog

## [0.3.0](https://github.com/JoMe92/arciva-chroma-lab/compare/v0.2.1...v0.3.0) (2026-01-13)


### Features

* 43 arbitrary free crop ([fbc9a38](https://github.com/JoMe92/arciva-chroma-lab/commit/fbc9a382bd41e44915444e68e47ccd405e2ed684))
* 44 flip mirror ([1cd9924](https://github.com/JoMe92/arciva-chroma-lab/commit/1cd99245488feb59b45db6e2504386e081a73d8a))
* Add explicit rect cropping with UI overlay, highlights/shadows, and vertical/horizontal skew controls. ([2926817](https://github.com/JoMe92/arciva-chroma-lab/commit/29268179e78095b5cec4d3fa8c5376fcc4ca6244)), closes [#43](https://github.com/JoMe92/arciva-chroma-lab/issues/43)
* add function to calculate opaque crop for rotated images using largest interior rectangle geometry ([d48e758](https://github.com/JoMe92/arciva-chroma-lab/commit/d48e758675cd73b156e21aa980da10eb7054cd31))
* Add horizontal and vertical image flipping to geometry adjustments. ([2641d0a](https://github.com/JoMe92/arciva-chroma-lab/commit/2641d0a6a4245c51ca5326221676a14e9755e658))
* add horizontal and vertical image flipping to shaders and renderer settings ([e85a014](https://github.com/JoMe92/arciva-chroma-lab/commit/e85a0146bfc56b94232e0bd9376ccb5ff651a899)), closes [#44](https://github.com/JoMe92/arciva-chroma-lab/issues/44)
* add sharpen, clarity, and dehaze effects ([6050957](https://github.com/JoMe92/arciva-chroma-lab/commit/60509575201c7659c27580903456ac91cdfb1fb9))
* add split toning support ([3166707](https://github.com/JoMe92/arciva-chroma-lab/commit/3166707f11593a24aeecb3af98c60c280065ecc7)), closes [#35](https://github.com/JoMe92/arciva-chroma-lab/issues/35)
* add support for .3dl and .xmp LUT formats ([4a56034](https://github.com/JoMe92/arciva-chroma-lab/commit/4a56034d56484e79841562fb5f7aa32f116becbf))
* add targeted HSL adjustments for 8 color ranges ([b72168a](https://github.com/JoMe92/arciva-chroma-lab/commit/b72168a58dd2ba8e57bb6b6b6a7b45e1de892587))
* add targeted HSL adjustments for 8 color ranges ([3dda187](https://github.com/JoMe92/arciva-chroma-lab/commit/3dda1875c68513b5196be18db13bb85be494beae)), closes [#35](https://github.com/JoMe92/arciva-chroma-lab/issues/35)
* **distortion:** implement radial lens distortion correction ([3f2b60b](https://github.com/JoMe92/arciva-chroma-lab/commit/3f2b60bb72df485e1d29bc4188cbdadcf1ea6d11)), closes [#35](https://github.com/JoMe92/arciva-chroma-lab/issues/35)
* enhance curves functionality and improve sidebar layout ([4139321](https://github.com/JoMe92/arciva-chroma-lab/commit/4139321cb8bc28b5f02da931ab0628f814313cbb)), closes [#35](https://github.com/JoMe92/arciva-chroma-lab/issues/35)
* implement 3D LUT support for all backends ([755cfd5](https://github.com/JoMe92/arciva-chroma-lab/commit/755cfd5a4ccbf58f14bae210e26e5de0388f62f7)), closes [#50](https://github.com/JoMe92/arciva-chroma-lab/issues/50)
* implement custom curves supporting Master/RGB channels ([d23f752](https://github.com/JoMe92/arciva-chroma-lab/commit/d23f7520aa70f2e4956b4d75d301fb3dc842ae74)), closes [#35](https://github.com/JoMe92/arciva-chroma-lab/issues/35)
* implement real-time histogram ([1b52ec2](https://github.com/JoMe92/arciva-chroma-lab/commit/1b52ec20420490c9a3b260d55646b8f70fb57e95)), closes [#48](https://github.com/JoMe92/arciva-chroma-lab/issues/48)
* implement split toning adjustments ([9cfa016](https://github.com/JoMe92/arciva-chroma-lab/commit/9cfa016ee48eacdfccf11b0e76019696459f355d)), closes [#35](https://github.com/JoMe92/arciva-chroma-lab/issues/35)
* implement stateless pipeline and source image caching ([32b4074](https://github.com/JoMe92/arciva-chroma-lab/commit/32b4074cd3b3673e95ade2f9c0b1e0527ad94c81))
* implement tiled rendering for high-res exports ([b2d15aa](https://github.com/JoMe92/arciva-chroma-lab/commit/b2d15aa3f2f74bd3e5e0844f4c610e9877326a02)), closes [#52](https://github.com/JoMe92/arciva-chroma-lab/issues/52)
* implement vignette effect ([185df2a](https://github.com/JoMe92/arciva-chroma-lab/commit/185df2a17c2f4e5941c1ee9a4637a1ae02403079))
* improve sidebar layout and add curves intensity control ([db0ee09](https://github.com/JoMe92/arciva-chroma-lab/commit/db0ee0973382aea331c74a93f847477e8624696f)), closes [#35](https://github.com/JoMe92/arciva-chroma-lab/issues/35)
* Performance Architecture Refactor (Stateless & SIMD) ([b7c0f22](https://github.com/JoMe92/arciva-chroma-lab/commit/b7c0f22d0ce04766565989a5d4e9969a2b2ddd37))
* **renderer:** implement high-ISO noise reduction ([dc41438](https://github.com/JoMe92/arciva-chroma-lab/commit/dc414380c03ff0070ee892ddd715d7415397c492)), closes [#51](https://github.com/JoMe92/arciva-chroma-lab/issues/51)
* unify rendering pipeline and enhance viewer UI ([61a59d6](https://github.com/JoMe92/arciva-chroma-lab/commit/61a59d64e3b14a47a523dfcaba7caadef0659359)), closes [#35](https://github.com/JoMe92/arciva-chroma-lab/issues/35)


### Bug Fixes

* disable lut application when no lut data is loaded ([6acf30d](https://github.com/JoMe92/arciva-chroma-lab/commit/6acf30d4cb356ed68f5a3a882d9d22eb062ac792)), closes [#50](https://github.com/JoMe92/arciva-chroma-lab/issues/50)
* **lint:** resolve clippy and fmt issues ([4340a4e](https://github.com/JoMe92/arciva-chroma-lab/commit/4340a4eca7ba1abcdc28b6b56235c842f46f871f))
* **renderer:** code fix SettingsUniform fields in WebGPU backend ([c368369](https://github.com/JoMe92/arciva-chroma-lab/commit/c36836948ede218040f0514fd04b10179cca7cfc))
* resolve compilation errors in operations.rs ([0dfa9b7](https://github.com/JoMe92/arciva-chroma-lab/commit/0dfa9b7bc8f615630d6bff0526d6f45371393b21))
* resolve compilation errors in renderer ([bebc493](https://github.com/JoMe92/arciva-chroma-lab/commit/bebc49382a5c38f00ebfb1d199f273b022456019))
* resolve quickfix-renderer compilation errors and update uniforms ([5cf1d6f](https://github.com/JoMe92/arciva-chroma-lab/commit/5cf1d6f65634421e7da9225d56fbe463230cff82))


### Performance Improvements

* implement WASM SIMD optimizations for blur, HSL, and denoise ([63da529](https://github.com/JoMe92/arciva-chroma-lab/commit/63da52903bb74953cfda56993840f1e9383b01ac))
* optimize HSL shader execution with conditional uniform ([7d1c1b9](https://github.com/JoMe92/arciva-chroma-lab/commit/7d1c1b95abc2354b87d293fa5ffea3838807ba24))

## [0.2.1](https://github.com/JoMe92/chroma-lab/compare/v0.2.0...v0.2.1) (2025-12-06)


### Bug Fixes

* **quickfix-renderer:** correct relative import paths in worker build ([c7baaac](https://github.com/JoMe92/chroma-lab/commit/c7baaacd7df9164e00853fa1c49df3f86b177382))
* **renderer:** fix worker import path by resolving build errors ([9f0517c](https://github.com/JoMe92/chroma-lab/commit/9f0517c9b7973c05d896b1cd0a16b8fceb670127))

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
