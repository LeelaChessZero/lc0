#!/usr/bin/env node
/*
  Folder-based multi-net smoke test for the lc0 wasm/JS backend.

  Scans a directory for network files (.pb / .pb.gz) and runs the same UCI
  session (uci -> isready -> position -> go -> bestmove) against each one
  in turn, reporting which head types each net actually exercised -- WDL
  vs. classical value, MLH present or not -- so testing a batch of nets
  covering different architectures is just a matter of putting them all in
  one folder and running this once.

  Usage:
    node test-nets-dir.mjs <path-to-folder> [nodes]

  Example:
    mkdir nets
    cp ~/downloads/11248.pb.gz ~/downloads/T30.pb.gz ~/downloads/744204.pb.gz nets/
    node test-nets-dir.mjs ./nets 64
*/
import { readFile, readdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function usage() {
	console.error(`
Usage: node test-nets-dir.mjs <path-to-folder> [nodes]

  <path-to-folder>   Required. Directory containing .pb / .pb.gz network files.
                      Every matching file in it is tested, one by one.
  [nodes]             Optional. Node budget per test (default: 64).

Example:
  node test-nets-dir.mjs ./nets 64
`);
	process.exit(1);
}

/**
 * Runs one full UCI session against a single network. Returns a report
 * object rather than throwing, so the caller can continue testing the
 * remaining nets even if one fails. Only ever called inside the
 * subprocess spawned by runInSubprocess below, one net at a time.
 */
async function testOneNet(label, netPath, nodes) {
	const report = { label, netPath, pass: false, bestmove: null, hasWdl: false, hasMovesLeft: false, error: null };

	try {
		const buildDir = path.join(__dirname, "build");
		const Module = (await import(path.join(buildDir, "lc0.js"))).default;
		const ort = await import("onnxruntime-web");

		const bytes = new Uint8Array(await readFile(netPath));

		const inputQueue = [
			"uci",
			"setoption name UCI_ShowWDL value true",
			"setoption name UCI_ShowMovesLeft value true",
			"isready",
			"position startpos",
			`go nodes ${nodes}`,
		];
		globalThis.lc0web_get_line = () => {
			if (inputQueue.length) return inputQueue.shift();
			return new Promise(() => {});
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
		await new Promise((resolve, reject) => {
			const timeout = setTimeout(() => reject(new Error("timed out after 120s")), 120000);

			Module({
				preRun: m => {
					emModule = m;
					const f = m.FS.open("net.pb.gz", "w");
					m.FS.write(f, bytes, 0, bytes.length);
					m.FS.close(f);
				},
				arguments: ["--preload", "-w", "net.pb.gz"],
				print: text => {
					if (text.includes(" wdl ")) report.hasWdl = true;
					if (text.includes(" movesleft ")) report.hasMovesLeft = true;
					if (text.startsWith("bestmove") && !sawBestmove) {
						sawBestmove = true;
						report.bestmove = text;
						report.pass = true;
						clearTimeout(timeout);
						resolve();
					}
				},
				printErr: () => {},
			}).catch(reject);
		});
	} catch (err) {
		report.error = err?.message ?? String(err);
	}

	return report;
}

// Each net gets a fresh Node subprocess -- see testOneNet's doc comment
// for why.
async function runInSubprocess(label, netPath, nodes) {
	const { fork } = await import("node:child_process");
	return new Promise(resolve => {
		const child = fork(fileURLToPath(import.meta.url), ["--worker", label, netPath, String(nodes)], {
			stdio: ["ignore", "ignore", "ignore", "ipc"],
		});
		let settled = false;
		const finish = report => {
			if (settled) return;
			settled = true;
			resolve(report);
		};
		child.on("message", finish);
		child.on("exit", code => {
			if (!settled) finish({ label, netPath, pass: false, error: `subprocess exited (code ${code}) without reporting` });
		});
	});
}

// ---- Entry point ------------------------------------------------------------

const args = process.argv.slice(2);

if (args[0] === "--worker") {
	// Re-invoked as a child process for a single net (see runInSubprocess).
	const [, label, netPath, nodesArg] = args;
	const report = await testOneNet(label, netPath, parseInt(nodesArg, 10));
	process.send(report);
	process.exit(0);
}

if (args.length < 1) usage();
const [folderArg, nodesArg] = args;
const nodes = nodesArg ? parseInt(nodesArg, 10) : 64;

const folder = path.resolve(folderArg);
let entries;
try {
	entries = await readdir(folder);
} catch (err) {
	console.error(`Could not read folder "${folder}": ${err.message}`);
	process.exit(1);
}

const netFiles = entries
	.filter(name => name.endsWith(".pb") || name.endsWith(".pb.gz"))
	.sort();

if (netFiles.length === 0) {
	console.error(`No .pb / .pb.gz files found in "${folder}".`);
	process.exit(1);
}

console.log(`Found ${netFiles.length} net(s) in ${folder}, testing at ${nodes} nodes each...\n`);

const reports = [];
for (const file of netFiles) {
	const netPath = path.join(folder, file);
	process.stdout.write(`  ${file} ... `);
	const report = await runInSubprocess(file, netPath, nodes);
	reports.push(report);
	console.log(report.pass ? "PASS" : `FAIL${report.error ? ` (${report.error})` : ""}`);
}

console.log("\n" + "=".repeat(70));
console.log("Summary");
console.log("=".repeat(70));
for (const r of reports) {
	console.log(`\n${r.label}`);
	console.log(`  result:      ${r.pass ? "PASS" : "FAIL"}`);
	if (r.pass) {
		console.log(`  bestmove:    ${r.bestmove}`);
		console.log(`  WDL head:    ${r.hasWdl ? "yes" : "no (classical value)"}`);
		console.log(`  MLH head:    ${r.hasMovesLeft ? "yes" : "no"}`);
	} else {
		console.log(`  error:       ${r.error}`);
	}
}

const failures = reports.filter(r => !r.pass).length;
console.log(`\n${reports.length - failures}/${reports.length} passed.`);
process.exit(failures);
