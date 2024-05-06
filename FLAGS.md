# Modes

`lc0` supports several operating modes, each of which has a different set of 
command line flags (although there are common ones).

Currently, `lc0` has the following modes:

| Mode |   Description |
|------|---------------|
| uci (*default*) | Acts as a UCI chess engine |
| selfplay | Plays one or multiple games with itself and optionally generates training data |
| debug | Generates debug data for a position |

To run `lc0` in any of those modes, specify a mode name as the first argument (`uci` may be omitted).
For example:

```bash
$ ./lc0 selfplay ...    # For selfplay mode
$ ./lc0 ...             # For UCI mode
```

In any of those modes, it's possible to get help using the `--help` command line argument:

```bash
$ ./lc0 --help            # Help for UCI mode
$ ./lc0 selfplay --help   # Help for selfplay mode
```

## UCI Mode

In UCI engine mode, all of command line parameters can also be changed using a UCI parameter.

List of command line flags:

| Flag | UCI Parameter | Description |
|------|---------------|-------------|
| -w PATH,<br>--weights=PATH | Network weights file path | Path to load network weights from.<br>Default is `<autodiscover>`, which makes it search for the latest (by file date) file in ./ and ./weights/ subdirectories which look like weights. |
| -t NUM,<br>--threads=NUM | Number of worker threads | Number of (CPU) threads to use.<br> Default is `2`. There's currently no use of making it more than 3 as it's limited by mutex contention which is yet to be optimized. |
| --nncache=SIZE | NNCache size | Number of positions to store in cache.<br>Default: `2000000` |
| <nobr>--backend=BACKEND</nobr><br><nobr>--backend-opts=OPTS</nobr> | NN backend to use<br>NN backend parameters | Configuration of backend parameters. Described in details [here](#backendconfiguration).<br>Default depends on particular build type (cuDNN, tensorflow, etc.). |
| --slowmover=NUM | Scale thinking time | Parameter value `X` means that the whole remaining time is split in such a way that the current move gets `X × Y` seconds, and next moves will get `1 × Y` seconds. However, due to smart pruning, the engine usually doesn't use all allocated time.<br>Default: `2.2`|
| <nobr>--move-overhead=NUM</nobr> | Move time overhead in milliseconds | How much overhead should the engine allocate for every move (to counteract things like slow connection, interprocess communication, etc.).<br>Default: `100` ms. |
| <nobr>--minibatch-size=NUM</nobr> | Minibatch size for NN inference | How many positions the engine tries to batch together for computation. Theoretically larger batches may reduce strengths a bit, especially on a small number of playouts.<br>Default is `256`. Every backend/hardware has different optimal value (e.g., `1` if batching is not supported). |
| <nobr>--max-prefetch=NUM</nobr> | Maximum prefetch nodes per NN call | When the engine can't gather a large enough batch for immediate use, try to prefetch up to `X` positions, which are likely to be useful soon, and put them in the cache.<br>Default: `32`. |
| <nobr>--cpuct=NUM</nobr> | Cpuct MCTS option | C_puct constant from Upper Confidence Tree search algorithm. Higher values promote more exploration/wider search, lower values promote more confidence/deeper search.<br>Default: `1.2`. |
| <nobr>--temperature=NUM</nobr> | Initial temperature | Tau value from softmax formula. If equal to 0, the engine also picks the best move to make. Larger values increase randomness while making the move.<br>Default: `0` |
| <nobr>--tempdecay-moves=NUM</nobr> | Moves with temperature decay | Reduce temperature for every move linearly from initial temperature to `0`, during this number of moves since the game started. `0` disables temperature decay.<br>Default: `0` |
| -n,<br>--[no-]noise | Add Dirichlet noise to root node | Add noise to root node prior probabilities. This allows the engine to explore moves which are known to be very bad, and this is useful to discover new ideas during training.<br>Default: `false` |
| <nobr>--[no-]verbose-move-stats | Display verbose move stats | Display `Q`, `V`, `N`, `U` and `P` values of every move candidate after each move.<br>Default: `false` |
| --[no-]smart-pruning  | Enable smart pruning | Default: `true` |
| --virtual-loss-bug=NUM | Virtual loss bug | Default: `0` |
| --fpu-reduction=NUM | First Play Urgency reduction | Default: `0.2` |
| --cache-history-length=NUM | The length of history to include in the cache | Default: `7` |
| --extra-virtual-loss=NUM | Extra virtual loss | Default: `0` |
| -l,<br>--logfile=FILENAME | Do debug logging into a file | Default is off (empty string) |


## Configuration Files
`lc0` supports using a configuration file instead of passing flags on the command line.  The default configuration file is `lc0.config`, but it can be changed with the `--config` command line flag.  `lc0` configuration files only support the long flags that begin with `--`, and there must only be 1 flag per line.  For example:
```
# Lines beginning with a # is a comment
--threads=1
--minibatch-size=32
--sticky-checkmate
# The -- is optional.  The following flags will work as well:
weights=10445.txt.gz
syzygy-paths=syzygy
logfile=lc0.log
```
You can tell `lc0` to ignore the default configuration file by passing `--config=` on the command line.  Command line arguments will override any arguments that also exist in the configuration file.


## Backend Configuration

To be explained. That's the most interesting and undocumented!


## Selfplay Mode

To be explained.


## Debug Mode

To be explained.
