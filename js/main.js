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

function Stream(worker, type, finished)
{
	let gotLine
	const lines = []
	
	worker.addEventListener("message", ({data}) =>
	{
		if (data.type !== type) return
		if (!gotLine) {
			lines.push(data.text)
			return
		}
		gotLine(data.text)
		gotLine = undefined
	})
	
	function next()
	{
		if (lines.length !== 0) return lines.shift()
		else return new Promise(resolve => gotLine = resolve)
	}
	
	const it = {next: async () => finished() && lines.length === 0 ? {done: true} : {done: false, value: await next()}}
	Object.freeze(it)
	
	const peek = () => lines[0]
	return {next, peek, [Symbol.asyncIterator]: () => it}
}

export function Lc0(network)
{
	const worker = new Worker(new URL("worker.js", import.meta.url), {type: "module"})
	
	let commands = []
	let post0 = command => commands.push(command)
	
	worker.addEventListener("message", () =>
	{
		worker.postMessage({network}, [network])
		for (const command of commands) worker.postMessage(command)
		commands = undefined
		post0 = command => worker.postMessage(command)
	}, {once: true})
	
	const post = command =>
	{
		if (finished) throw new Error("Cannot post command to finished Lc0")
		post0(String(command))
	}
	
	let finished = false
	
	// todo: this should send a message to the worker instead
	// so that it can end its pthread workers too
	function finish()
	{
		finished = true
		worker.terminate()
	}
	
	const stdout = Stream(worker, "stdout", () => finished)
	const stderr = Stream(worker, "stderr", () => finished)
	
	const lc0 = {post, finish, ...stdout, stderr, get finished() { return finished }}
	Object.freeze(lc0)
	return lc0
}
