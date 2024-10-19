import Module from "./build/lc0.js"
import {InferenceSession, Tensor} from "onnxruntime-web/all"

let gotLine
let lines = []

addEventListener("message", ({data}) =>
{
	if (typeof data !== "string") return
	if (!gotLine) {
		lines.push(data)
		return
	}
	gotLine(data)
	gotLine = undefined
})

let {data: {network}} = await new Promise(resolve =>
{
	postMessage({type: "ready"})
	addEventListener("message", resolve, {once: true})
})

let session = await InferenceSession.create(await new Response(network).arrayBuffer(), {executionProviders: ["webgpu", "wasm"]})
let map = new Map()

let lc0web_get_line = async () =>
{
	if (lines.length !== 0) return lines.shift()
	return new Promise(resolve => gotLine = resolve)
}

let lc0web_is_cpu = () => true

let id = 0
let lc0web_id = () =>
{
	let i = id++
	map.set(i, {input: []})
	return i
}

let lc0web_batch_size = id => map.get(id).input.length

let lc0web_remove = id => map.delete(id)

let lc0web_q_val = (id, sample) =>
{
	let [w, d, l] = map.get(id).output["/output/wdl"].cpuData
	return w - l
}

let lc0web_d_val = (id, sample) =>
{
	let [w, d, l] = map.get(id).output["/output/wdl"].cpuData
	return d
}

let lc0web_p_val = (id, sample, moveID) =>
	map.get(id).output["/output/policy"].cpuData[sample * 1858 + moveID]

let lc0web_m_val = (id, sample) =>
	map.get(id).output["/output/mlh"].cpuData[sample]

let lc0web_add_input = id => map.get(id).input.push([])

let lc0web_add_plane = (id, index, mask, value) =>
{
	let array = map.get(id).input[index]
	for (let i = 0 ; i < 64 ; i++) {
		if (mask & 1n) array.push(value)
		else array.push(0)
		mask >>= 1n
	}
}

let lc0web_compute = async id =>
{
	let value = map.get(id)
	let {input} = value
	let array = new Float32Array(input.flat(Infinity))
	let tensor = new Tensor("float32", array, [input.length, 112, 8, 8])
	value.output = await session.run({"/input/planes": tensor})
}

Object.assign(globalThis, {
	lc0web_get_line,
	lc0web_is_cpu,
	lc0web_id,
	lc0web_batch_size,
	lc0web_remove,
	lc0web_q_val,
	lc0web_d_val,
	lc0web_p_val,
	lc0web_m_val,
	lc0web_add_input,
	lc0web_add_plane,
	lc0web_compute,
})

Module({
	arguments: ["--preload"],
	print: text => postMessage({type: "stdout", text}),
	printErr: text => postMessage({type: "stderr", text}),
})
