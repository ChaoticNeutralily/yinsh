# TODO

- [x] make `play_yinsh_with_visuals.py` more modular instead of one big function
- [x] make random_bot take full gamestate so `play_yinsh_with_visuals.py` is bot-generic
- [x] prints initialization for final game state when gui game is closed.
- [x] implement non-gui play that tabulates results
- [ ] cythonize yinsh.py
- [x] bug in tabulation or runner. never draws and all the games are identical result in loop
  - due to yinsh game kept persisting
  - even reinitializing game didn't fix
  - it was default args bug but still happened with the default_factory lambda func which was tricky.
- [ ] implement Elo
- [ ] make manual features
- [ ] make simple bots
- [ ] copy floyd bot
- [ ] copy other python bots if they're simple enough to directly copy
