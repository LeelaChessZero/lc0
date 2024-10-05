/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

import Module from "./build/lc0.js"
import {InferenceSession, Tensor} from "onnxruntime-web/all"

let gotLine
const lines = []

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

const {data: {network}} = await new Promise(resolve =>
{
	postMessage({type: "ready"})
	addEventListener("message", resolve, {once: true})
})

let bytes = new Uint8Array(await new Response(network).arrayBuffer())

let id = 0
const map = new Map()

function lc0web_get_line()
{
	if (lines.length !== 0) return lines.shift()
	return new Promise(resolve => gotLine = resolve)
}

function lc0web_is_cpu(_id)
{
	// TODO
	return true
}

function lc0web_computation(id2)
{
	const i = id++
	map.set(i, {input: [], session: map.get(id2)})
	return i
}

function lc0web_batch_size(id)
{
	return map.get(id).input.length
}

function lc0web_remove(id)
{
	return map.delete(id)
}

function lc0web_q_val(id, _sample)
{
	const [w, _d, l] = map.get(id).output["/output/wdl"].cpuData
	return w - l
}

function lc0web_d_val(id, _sample)
{
	const [_w, d] = map.get(id).output["/output/wdl"].cpuData
	return d
}

function lc0web_p_val(id, sample, moveID)
{
	return map.get(id).output["/output/policy"].cpuData[sample * 1858 + moveID]
}

function lc0web_m_val(id, sample)
{
	return map.get(id).output["/output/mlh"].cpuData[sample]
}

function lc0web_add_input(id)
{
	return map.get(id).input.push([])
}

function lc0web_add_plane(id, index, mask, value)
{
	const array = map.get(id).input[index]
	for (let i = 0 ; i < 64 ; i++) {
		if (mask & 1n) array.push(value)
		else array.push(0)
		mask >>= 1n
	}
}

async function lc0web_compute(id)
{
	const value = map.get(id)
	const array = new Float32Array(value.input.flat(Infinity))
	const tensor = new Tensor("float32", array, [value.input.length, 112, 8, 8])
	value.output = await value.session.run({"/input/planes": tensor})
}

async function lc0web_network(data, length)
{
	const i = id++
	const buffer = module.HEAPU8.subarray(data, data + length)
	const session = await InferenceSession.create(buffer, {executionProviders: ["webgpu", "wasm"]})
	map.set(i, session)
	return i
}

Object.assign(globalThis, {
	lc0web_get_line,
	lc0web_is_cpu,
	lc0web_computation,
	lc0web_batch_size,
	lc0web_remove,
	lc0web_q_val,
	lc0web_d_val,
	lc0web_p_val,
	lc0web_m_val,
	lc0web_add_input,
	lc0web_add_plane,
	lc0web_compute,
	lc0web_network,
})

let module
Module({
	preRun: m =>
	{
		module = m
		const file = module.FS.open("net.pb.gz", "w")
		module.FS.write(file, bytes, 0, bytes.length)
		module.FS.close(file)
		// free the buffer
		if (bytes.buffer.transfer) bytes.buffer.transfer(0)
		// let it be garbage-collected
		bytes = undefined
	},
	arguments: ["--preload", "-w", "net.pb.gz"],
	print: text => postMessage({type: "stdout", text}),
	printErr: text => postMessage({type: "stderr", text}),
})
