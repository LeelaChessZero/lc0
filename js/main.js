let Stream = (worker, type, finished) =>
{
	let gotLine
	let lines = []
	
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
	
	let next = () =>
	{
		let value
		if (lines.length !== 0) return lines.shift()
		else return new Promise(resolve => gotLine = resolve)
	}
	
	let it = {next: async () => finished() && lines.length === 0 ? {done: true} : {done: false, value: await next()}}
	Object.freeze(it)
	
	let peek = () => lines[0]
	
	return {next, peek, [Symbol.asyncIterator]: () => it}
}

export let Lc0 = network =>
{
	let worker = new Worker(new URL("worker.js", import.meta.url), {type: "module"})
	
	let commands = []
	let post0 = command => commands.push(command)
	
	worker.addEventListener("message", () =>
	{
		worker.postMessage({network}, [network])
		for (let command of commands) worker.postMessage(command)
		commands = undefined
		post0 = command => worker.postMessage(command)
	}, {once: true})
	
	let post = command =>
	{
		if (finished) throw new Error("Cannot post command to finished Lc0")
		post0(String(command))
	}
	
	let finished = false
	
	// todo: this should send a message to the worker instead
	// so that it can end its pthread workers too
	let finish = () =>
	{
		finished = true
		worker.terminate()
	}
	
	let stdout = Stream(worker, "stdout", () => finished)
	let stderr = Stream(worker, "stderr", () => finished)
	
	let lc0 = {post, finish, ...stdout, stderr, get finished() { return finished }}
	Object.freeze(lc0)
	return lc0
}
