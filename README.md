# A pygame implementation of the game yinsh
Yinsh is a two player abstract strategy game by Kris Burm as part of the project GIPF.
https://www.gipf.com/yinsh/
Here, I have a pygame implementation of a visual interface to play yinsh either with two players using a mouse, one player using a mouse against a bot, or to watch two bots play against one another.
<p align="center">
  <img width="506" alt="A screenshot of the pygame yinsh implementation during a game with moves for player 1 highlighted. " src="https://github.com/ChaoticNeutralily/yinsh/assets/156118924/f00fff10-ee86-40fa-b8a3-a96cd0591cdc">
</p>

My initial implementation of the game logic is a python port of the haskell implementation by David Peter at https://github.com/sharkdp/yinsh
This repo also has various yinsh bots I've been working on to practice making game playing agents.
The initial tree search bots also use ports of the heuristics on David Peter's version.
I've since started adding more heuristics, and am currently working on MCTS based bots, and neural net value/policy bots.

# TODO and what's done

- [x] make `play_yinsh_with_visuals.py` more modular instead of one big function
- [x] make random_bot take full gamestate so `play_yinsh_with_visuals.py` is bot-generic
- [x] prints initialization for final game state when gui game is closed.
- [x] implement non-gui play that tabulates results
- [ ] cythonize yinsh.py
- [x] bug in tabulation or runner. never draws and all the games are identical result in loop
  - due to yinsh game kept persisting
  - even reinitializing game didn't fix
  - it was default args bug but still happened with the default_factory lambda func which was tricky.
- [x] implement performance Elo
- [x] implement glicko2
  - [ ] to run correctly, requires saving separate win/loss/draw arrays for current scoring timeframe versus all time.
- [x] implement I/O for automatic ranking updates. 
- [ ] have the visual version keep track of gamestates to allow undos for casual play
- [ ] implement time control
  - [ ] could just use python time library and just count up time used.
Or, more complicated use
  - [ ] server
  - [ ] client
- [ ] make game menu to optionally use instead of just commandline args
  - [ ] allow to pick which player is human/bot
  - [ ] set delay for bot(s)
  - [ ] set the time control
  - [ ] set colors
  - [ ] track rankings
  - [ ] leaderboard
  - [ ] optionally show heatmap of choice of bot's values instead of equal highlighting for all valid moves
  - [ ] turn valid move highlighting on/off
- [x] alphabeta negamax implementation
  - [x] make manual features/heuristics for non-terminal evaluation
  - [x] make simple bots
  - [x] implement floyd bot from David Peter's site
    - [x] find cause of difference in floyd behavior during openings.
      - function to get move values didn't actually make the move it was checking, so all moves valued the same board state. big oof
  - [ ] transposition table
    - [ ] rewrite the floyd heuristics to just take a board instance
  - [ ] hp tuning / BO to find best weight of basic heuristics and apply the weighted sum as a bot
  - [ ] more heuristics for alphabeta, like from the iit student's bots
  - [ ] principal variation search / negascout
  - [ ] smarter move ordering
- [ ] mcts bots (trying something like alphazero for these)
  - [x] game_state -> flat tensor
  - [x] flat tensor -> value and move probability
  - [x] decode prob over board into actual move probabilities for valid moves
  - [ ] gamestate -> 2d tensor stack
  - [ ] conv/resnet: 2d tensor stack -> value and move probability
  - [ ] Do gnns work better for these than flat arrays?
  - [x] mcts code
  - [x] training code for policy improvement using mcts
  - [ ] use nn value functions for alphabeta
  - [ ] use nn policy directly for actual playing
  - [ ] mcts using value for actual playing.
- [ ] implement more heuristics
