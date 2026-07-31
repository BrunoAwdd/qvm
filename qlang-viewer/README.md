# QLang Viewer

React/Vite circuit viewer backed by the shared Rust QLang parser compiled to WebAssembly.

```bash
cd ../qvm
wasm-pack build qlang-wasm --target web \
  --out-dir ../../qlang-viewer/src/wasm --out-name qlang_wasm
cd ../qlang-viewer
pnpm install
pnpm dev
```

`pnpm build` performs the TypeScript and production Vite build. The viewer statically projects gates from the real QLang AST. It reports parser errors and warns when dynamic expressions or control flow cannot be represented as one deterministic circuit.
