#!/usr/bin/env node
/*
  Headless UCI smoke test for the lc0 wasm build.

  Drives build/lc0.js + build/lc0.wasm through a real UCI session (uci ->
  isready -> position -> go -> bestmove) without needing a browser, by
  wiring up the same lc0web_* global functions that js/worker.js provides
  at runtime -- using onnxruntime-web directly for NN inference.

  Usage:
    npm install                 # installs onnxruntime-web (devDependency)
    ./build.sh                  # produces build/lc0.js, build/lc0.wasm
    node test-engine.mjs <path-to-net.pb.gz-or-.pb> [nodes]

  Example:
    node test-engine.mjs ~/nets/maia-1100.pb.gz 64

  Exit code 0 on a successful bestmove, non-zero on failure/timeout.
*/
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function usage() {
	console.error(`
Usage: node test-engine.mjs <path-to-net.pb.gz-or-.pb> [nodes]

  <path-to-net>   Required. Path to an lc0 weights file (.pb or .pb.gz).
  [nodes]         Optional. Node budget for the test "go" command (default: 64).

Examples:
  node test-engine.mjs ~/nets/maia-1100.pb.gz
  node test-engine.mjs ~/nets/maia-1100.pb.gz 128
`);
	process.exit(1);
}

async function runUciSmokeTest(netPath, nodes) {
	const buildDir = path.join(__dirname, "build");
	const Module = (await import(path.join(buildDir, "lc0.js"))).default;
	const ort = await import("onnxruntime-web");

	const bytes = new Uint8Array(await readFile(netPath));

	const inputQueue = ["uci", "isready", "position startpos", `go nodes ${nodes}`];
	globalThis.lc0web_get_line = () => {
		if (inputQueue.length) return inputQueue.shift();
		return new Promise(() => {}); // no more scripted input; let the timeout handle it
	};

	let id = 0;
	const sessions = new Map();
	let emModule;

	globalThis.lc0web_is_cpu = () => true;
	globalThis.lc0web_computation = netId => {
		const i = id++;
		sessions.set(i, { input: [], session: sessions.get(netId) });
		return i;
	};
	globalThis.lc0web_batch_size = i => sessions.get(i).input.length;
	globalThis.lc0web_remove = i => sessions.delete(i);

	const out = (i, name) => sessions.get(i).output[name];
	const data = t => t.cpuData ?? t.data;

	globalThis.lc0web_q_val = (i, sample) => {
		const wdl = out(i, "/output/wdl");
		if (wdl) { const d = data(wdl); return d[sample * 3] - d[sample * 3 + 2]; }
		return data(out(i, "/output/value"))[sample];
	};
	globalThis.lc0web_d_val = (i, sample) => {
		const wdl = out(i, "/output/wdl");
		return wdl ? data(wdl)[sample * 3 + 1] : 0;
	};
	globalThis.lc0web_p_val = (i, sample, moveId) => data(out(i, "/output/policy"))[sample * 1858 + moveId];
	globalThis.lc0web_m_val = (i, sample) => {
		const mlh = out(i, "/output/mlh");
		return mlh ? data(mlh)[sample] : 0;
	};
	globalThis.lc0web_add_input = i => sessions.get(i).input.push([]);
	globalThis.lc0web_add_plane = (i, index, mask, value) => {
		const array = sessions.get(i).input[index];
		for (let b = 0; b < 64; b++) { array.push(mask & 1n ? value : 0); mask >>= 1n; }
	};
	globalThis.lc0web_compute = async i => {
		const value = sessions.get(i);
		const array = new Float32Array(value.input.flat(Infinity));
		const tensor = new ort.Tensor("float32", array, [value.input.length, 112, 8, 8]);
		value.output = await value.session.run({ "/input/planes": tensor });
	};
	globalThis.lc0web_network = async (dataPtr, length) => {
		const i = id++;
		const buffer = emModule.HEAPU8.slice(dataPtr, dataPtr + length);
		const session = await ort.InferenceSession.create(buffer, { executionProviders: ["wasm"] });
		sessions.set(i, session);
		return i;
	};

	let sawBestmove = false;
	const result = await new Promise((resolve, reject) => {
		const timeout = setTimeout(() => reject(new Error("Timed out waiting for bestmove (120s)")), 120000);

		Module({
			preRun: m => {
				emModule = m;
				const f = m.FS.open("net.pb.gz", "w");
				m.FS.write(f, bytes, 0, bytes.length);
				m.FS.close(f);
			},
			arguments: ["--preload", "-w", "net.pb.gz"],
			print: text => {
				console.log("lc0>", text);
				if (text.startsWith("bestmove") && !sawBestmove) {
					sawBestmove = true;
					clearTimeout(timeout);
					resolve(text);
				}
			},
			printErr: text => console.error("lc0!", text),
		}).catch(reject);
	});

	return result;
}

// ---- Entry point ------------------------------------------------------------

const positional = process.argv.slice(2);
if (positional.length < 1) usage();
const [netPath, nodesArg] = positional;
const nodes = nodesArg ? parseInt(nodesArg, 10) : 64;

console.log(`Running a live UCI session (net: ${netPath}, nodes: ${nodes})\n`);
try {
	const bestmove = await runUciSmokeTest(netPath, nodes);
	console.log(`\n\u2713 PASS -- engine responded: "${bestmove}"`);
	process.exit(0);
} catch (err) {
	console.error(`\n\u2717 FAIL -- ${err.message ?? err}`);
	process.exit(1);
}
