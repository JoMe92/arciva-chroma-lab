# Rust Project Compass 🦀

This guide describes the typical structure of a Rust project, common conventions, and the recommended workflow. It helps you navigate Rust codebases quickly and write idiomatic ("Rustacean") code.

## 1. Directory Structure (Standard Layout)

Rust projects (created with `cargo new`) follow a strict convention. This makes it easy to orient yourself in unfamiliar projects.

### Base Directories

| Path | Description |
|------|-------------|
| `Cargo.toml` | **The Project Manifesto.** Contains metadata (name, version), dependencies, and workspace definitions. |
| `Cargo.lock` | Automatically generated. Freezes the exact versions of all dependencies to guarantee reproducible builds. **Do not edit manually!** |
| `src/` | The actual source code. |
| `target/` | All compilation artifacts (binaries, libraries) land here. Usually becomes very large and should be in `.gitignore`. |

### Special Directories (Optional, but important)

| Path | Purpose | Command to Run |
|------|---------|----------------|
| `tests/` | **Integration Tests.** Every file here (`*.rs`) is compiled as an *external crate*. They test only the public API of your library, exactly as a user would use it. | `cargo test` |
| `examples/` | **Example Programs.** Show users how to use the library. Very valuable for documentation. | `cargo run --example <name>` |
| `benches/` | **Benchmarks.** Performance tests (often using `criterion`). Measure how fast code runs and if changes slow it down. | `cargo bench` |

---

## 2. Code Organization in `src/`

### Libraries vs. Binaries

* **`src/lib.rs`**: The entry point for a **Library**. Here you define which modules are public (`pub mod`).
* **`src/main.rs`**: The entry point for an **Application**. Contains `fn main()`.
  * *Best Practice:* Keep `main.rs` thin! Put most logic into `lib.rs` so it is testable and reusable. `main.rs` should often only parse arguments and call `lib::run()`.

### Modules (`mod`)

Rust uses an explicit module system. A file `my_mod.rs` is not automatically found. It must be registered:

```rust
// in lib.rs or main.rs
mod my_mod; // Looks for my_mod.rs or my_mod/mod.rs
```

### Tests in `src/` (Unit Tests)

Unlike many other languages, **Unit Tests live directly in the file** they test.

**Why?** So they can also test *private* functions.

```rust
// src/my_calculation.rs

fn add_private(a: i32, b: i32) -> i32 {
    a + b
}

// Convention: Dedicated 'tests' module with #[cfg(test)]
#[cfg(test)]
mod tests {
    use super::*; // Imports everything from parent module

    #[test]
    fn test_add_private() {
        assert_eq!(add_private(2, 2), 4);
    }
}
```

---

## 3. Rust Specifics & Idioms

### Documentation is Code

Comments with `///` (three slashes) are documentation comments. They support Markdown.

The best part: Code blocks in docs are **executed as tests**!

```rust
/// Adds one to the number.
///
/// # Examples
/// ```
/// let x = 5;
/// assert_eq!(my_crate::add_one(x), 6);
/// ```
pub fn add_one(x: i32) -> i32 { x + 1 }
```

Generate docs locally with: `cargo doc --open`

### Workspaces

For large projects (Monorepos), you use Workspaces. A central `Cargo.toml` in the root manages multiple sub-projects (Crates).

* **Benefit:** Shared `target` folder (saves space & time) and shared `Cargo.lock`.
* Structure:

    ```
    root/
      Cargo.toml ([workspace] members = ["backend", "frontend"])
      backend/
        Cargo.toml
        src/
      frontend/
        Cargo.toml
        src/
    ```

### Feature Flags

In `Cargo.toml`, you can define `[features]`. This allows users to toggle parts of the library on or off to reduce binary size or dependencies.

---

## 4. The "Holy Grail" of Tooling

Rust developers rely extremely on standard tools. Use them to go with the flow, not against it.

### 1. `rustfmt` (The Formatter)

Never discuss spaces or brackets.

* **Command:** `cargo fmt`
* Makes code automatically conform to the standard style guide.

### 2. `clippy` (The Linter)

Your strict but helpful teacher. Finds bugs and bad style.

* **Command:** `cargo clippy`
* **Rule:** Do not commit if Clippy output warnings! Clippy teaches you how to write "idiomatic" Rust.

### 3. `cargo check`

Does not finish compiling code, just checks for errors.

* **Command:** `cargo check`
* Much faster than `cargo build`. Use this constantly while writing code.

### 4. `cargo test`

Executes everything:

* Unit Tests in `src/`
* Integration Tests in `tests/`
* Doc Tests in comments

## Workflow Summary

1. Write code.
2. `cargo check` (quick feedback if it compiles).
3. `cargo clippy` (Check for style and potential bugs).
4. `cargo fmt` (Cleanup).
5. `cargo test` (Ensure nothing is broken).
