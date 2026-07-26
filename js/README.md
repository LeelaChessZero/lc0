Leela Chess Zero (Wasm)
===

Lc0 compiled to WebAssembly, running natively in the browser via a Web
Worker -- no server-side engine required. Neural network inference runs
through [onnxruntime-web] (WebGPU when available, falling back to
multi-threaded wasm/SIMD otherwise); the network is converted from lc0's
own `.pb`/`.pb.gz` weights format to ONNX inside the engine itself, so any
standard lc0 net works without a separate conversion step.

[onnxruntime-web]: <https://github.com/microsoft/onnxruntime>


Compiling
---

Prerequisites:

- [Emscripten] -- **pin to 3.1.64.** Newer emscripten versions changed how
  `-pthread` builds emit code, causing some of them to declare extra
  `wasi_snapshot_preview1` imports (`clock_time_get`, `proc_exit`) that
  Firefox and Chrome reject with `LinkError: ... function import requires
  a callable` at load time (Edge has been observed to tolerate it; do not
  rely on that). This is a known upstream regression --
  see [emscripten#18396]. 3.1.64 is confirmed to build cleanly with none of
  these extra imports.
  ```sh
  ./emsdk install 3.1.64
  ./emsdk activate 3.1.64
  source ./emsdk_env.sh
  em++ --version   # should print 3.1.64
  ```
- [npm]
- [Meson]

Then, from this directory:

```sh
npm install
npm run build   # or: ./build.sh
```

This produces `build/lc0.js` and `build/lc0.wasm` (plus a couple of
smaller pthread-bootstrap files). `npm run pack` additionally bundles
`main.js`/`worker.js` with esbuild into `dist/` and tars it up as
`lc0.tar` -- the shape you'd `npm install` as a package elsewhere.

[Emscripten]: <https://emscripten.org/docs/getting_started/downloads.html>
[emscripten#18396]: <https://github.com/emscripten-core/emscripten/issues/18396>
[npm]: <https://docs.npmjs.com/cli/configuring-npm/install>
[Meson]: <https://mesonbuild.com/Getting-meson.html>


Testing
---

`test-engine.mjs` drives a real UCI session against the compiled engine
from plain Node -- no browser needed. It wires up the same `lc0web_*`
global functions `worker.js` provides at runtime and uses
onnxruntime-web directly for inference.

```sh
node test-engine.mjs <path-to-net.pb.gz-or-.pb> [nodes]

# e.g.
node test-engine.mjs ~/nets/maia-1100.pb.gz 64
```

Exits 0 on a successful `bestmove`, non-zero on failure or a 120s timeout.
Useful both as a fast local sanity check after a build and for CI.

```sh
mkdir nets
cp ~/downloads/11248.pb.gz ~/downloads/T30.pb.gz ~/downloads/744204.pb.gz nets/
node test-nets-dir.mjs ./nets 64
```
 
It picks up every `.pb`/`.pb.gz` file in the given folder, tests each one
the same way `test-nets.mjs` does (own subprocess, WDL/MLH detection,
pass/fail summary), and exits 0 only if every net in the folder passes.
Usage
---

```js
import { Lc0 } from "lc0"   // or "./main.js", relative to this directory

const network = await fetch("net.pb.gz").then(r => r.arrayBuffer())
const lc0 = Lc0(network)

lc0.post("uci")
lc0.post("isready")
lc0.post("position startpos")
lc0.post("go nodes 64")

for await (const line of lc0) {
	console.log(line)
	if (line.startsWith("bestmove")) break
}

lc0.finish()
```

`Lc0(network)` spawns the worker and returns:

- `post(command)` -- send a UCI command (string).
- `for await (const line of lc0)` / `lc0.next()` / `lc0.peek()` -- iterate
  engine stdout, one UCI line at a time.
- `lc0.stderr` -- a separate async-iterable stream for stderr output
  (diagnostics/warnings; not necessarily fatal).
- `lc0.finish()` -- terminate the worker.
- `lc0.finished` -- whether `finish()` has been called.

### Browser requirements

The engine is built with `-pthread`, so its wasm memory is a
`SharedArrayBuffer`. The page must be [cross-origin isolated] for that to
work: serve it with

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

`self.crossOriginIsolated` is fixed by whichever top-level HTTP navigation
loaded the current document -- a client-side route transition (in a
framework's SPA router) into a page with these headers, from a page that
lacked them, will *not* pick them up. If lc0 fails to start with a
`SharedArrayBuffer is not defined` error, check `self.crossOriginIsolated`
in devtools first; a hard reload of the current page (not a client-side
navigation) will fix it if that's the cause.

`main.js` loads `worker.js` via `new URL("worker.js", import.meta.url)`,
so it must be served alongside `main.js`. `worker.js` in turn imports
`lc0.js` from a `build/` subdirectory relative to itself
(`./build/lc0.js` -- see the layout `npm run pack` produces under
`dist/`), and `lc0.js` self-references `lc0.wasm`, and itself again if the
browser spawns additional pthread workers, both via
`new URL(..., import.meta.url)` relative to its own location. So the
directory structure matters: `lc0.js` and `lc0.wasm` must stay siblings of
each other (wherever you put them), even if `main.js`/`worker.js` move
elsewhere.

[cross-origin isolated]: <https://web.dev/articles/coop-coep>
