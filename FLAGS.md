# Modes

`lc0` supports several operating modes, each on which has different set of 
command line flags (although there are common ones).

Currently `lc0` has the following modes:

| Mode |   Description |
|------|---------------|
| uci *(default)* | Acts as UCI chess engine |
| selfplay | Plays one or multiple games with itself and optionally generates training data |
| debug | Generates debug data for a position |

To run `lc0` in any of those modes, specify a mode name as a first argument (`uci` may be omitted).
For example:

```bash
$ ./lc0 selfplay ...    # for selfplay mode
$ ./lc0 ...             # for uci mode
```

In any of those modes it's possible to get help using `--help` command line argument:

```bash
$ ./lc0 --help            # help for uci mode
$ ./lc0 selfplay --help   # help for selfplay mode
```

## UCI mode

In UCI engine mode all of command line parameters can also be changed using UCI parameter.

List of command line flags:

| Flag | Uci parameter | Description |
|------|---------------|-------------|
| -w PATH,<br>--weights=PATH | Network weights file path | Path to load network weights from.<br>Default is `<autodiscover>`, which makes it search for the latest (by file date) file in ./ and ./weights/ subdirectories which looks like weights. |
| -t NUM,<br>--threads=NUM | Number of worker threads | Number of (CPU) threads to use.<br> Default is `2`. There's no use of making it more than 3 currently as it's limited by mutex contention which is yet to be optimized. |
| --nncache=SIZE | NNCache size | Number of positions to store in cache.<br>Default: `200000` |
| <nobr>--backend=BACKEND</nobr><br><nobr>--backend-opts=OPTS</nobr> | NN backend to use<br>NN backend parameters | Configuration of backend parameters. Described in details [here](#backendconfiguration).<br>Default depends on particular build type (cuDNN, tensorflow, etc). |
| --slowmover=NUM | Scale thinking time | Parameter value X means that whole remaining time is split in such a way that current move gets X×Y seconds, and next moves will get 1×Y seconds. However, due to smart pruning, the engine usually doesn't use all allocated time.<br>Default: `2.2`|
| <nobr>--move-overhead=NUM</nobr> | Move time overhead in milliseconds | How much overhead should the engine allocate for every move (to counteract things like slow connection, interprocess communication, etc).<br>Default: `100`ms. |
| <nobr>--minibatch-size=NUM</nobr> | Minibatch size for NN inference | Now many positions the engine tries to batch together for computation. Theoretically larger batches may reduce strengths a bit, especially on small number of playouts.<br>Default is `256`. Every backend/hardware has different optimal value (e.g. `1` if batching is not supported). |
| <nobr>--max-prefetch=NUM</nobr> | Max prefetch nodes, per NN call | When engine cannot gather large enough batch for immediate use, try to prefetch up to X positions which are likely to be useful soon, and put them into cache.<br>Default: `32`. |
| <nobr>--cpuct=NUM</nobr> | Cpuct MCTS option | C_puct constant from "Upper confidence trees search" algorithm. Higher values promote more exploration/wider search, lower values promote more confidence/deeper search.<br>Default: `1.2`. |
| <nobr>--temperature=NUM</nobr> | Initial temperature | Tau value from softmax formula. If equal to 0, the engine also picks the best move to make. Larger values increase randomness while making the move.<br>Default: `0` |
| <nobr>--tempdecay-moves=NUM</nobr> | Moves with temperature decay | Reduce temperature for every move linearly from initial temperature to 0, during this number of moves since game start. `0` disables tempdecay.<br>Default: `0` |
| -n,<br>--[no-]noise | Add Dirichlet noise at root node | Add noise to root node prior probabilities. That allows engine to explore moves which are known to be very bad, which is useful to discover new ideas during training.<br>Default: `false` |
| <nobr>--[no-]verbose-move-stats | Display verbose move stats | Display Q, V, N, U and P values of every move candidate after each move.<br>Default: `false` |
| --[no-]smart-pruning  | Enable smart pruning | Default: `true` |
| --virtual-loss-bug=NUM | Virtual loss bug | Default: `0` |
| --fpu-reduction=NUM | First Play Urgency Reduction | Default: `0.2` |
| --cache-history-length=NUM | Length of history to include in cache | Default: `7` |
| --extra-virtual-loss=NUM | Extra virtual loss | Default: `0` |
| -l,<br>--debuglog=FILENAME | Do debug logging into file | Default if off. (empty string) |


## Backend configuration

To be written. That's the most interesting and undocumented!


## Selfplay mode

TBD

## Debug mode

TBD
