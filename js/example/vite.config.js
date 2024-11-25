export default {
	server: {
		headers: {
			"Cross-Origin-Embedder-Policy": "require-corp",
			"Cross-Origin-Opener-Policy": "same-origin",
		}
	},
	optimizeDeps: {
		exclude: ["lc0"],
	},
}
