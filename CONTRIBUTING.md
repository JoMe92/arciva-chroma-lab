# Contributing

## Coding Standards

- **Rust**: Follow standard Rust formatting and idioms.
  - Run `pixi run lint` before committing to ensure `cargo fmt` and `cargo clippy` pass.
- **TypeScript**: Use ESLint and Prettier (configured in example app).

## Testing

- Run `pixi run test` to execute Rust unit tests.
- Manual verification: Run `pixi run dev` and test the changes in the browser example.

## Building

- `pixi run build` will rebuild the WASM package.
- The output is located in `quickfix-renderer/pkg`.

## Workflow

1. Create a feature branch.
2. Make changes.
3. Run `pixi run lint` and `pixi run test`.
4. Submit a Pull Request.
