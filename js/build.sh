#!/usr/bin/env sh
set -ex
meson setup --buildtype=release -Ddefault_library=static --prefer-static --cross-file=../cross-files/wasm32-emscripten -Dblas=false build .. || :
meson compile -C build lc0
esbuild --minify --outdir=dist --format=esm main.js worker.js build/lc0.js build/lc0.worker.mjs
mv dist/build/lc0.worker.js dist/build/lc0.worker.mjs
cp build/lc0.wasm dist/build
cat > dist/package.json << END
{
	"name": "lc0",
	"description": "Leela Chess Zero",
	"version": "0.0.0.1",
	"license": "GPL",
	"homepage": "https://lczero.org",
	"repository": {
		"type": "git",
		"url": "https://github.com/LeelaChessZero/lc0"
	},
	"main": "./main.js",
	"exports": {
		".": {
			"import": "./main.js"
		}
	},
	"dependencies": {
		"onnxruntime-web": "1.19.2"
	}
}
END
