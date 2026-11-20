# Rating Your Moves website

Live site: https://jimhcyang.github.io/rating_your_moves/

A buildless static site with two views: a study of latent skill and a move-by-move chess example. GitHub Pages serves this directory from `main`.

Run a local preview from the repository root:

```sh
python -m http.server 8765 --directory docs
```

Open http://localhost:8765. Edit `index.html` for the study, `assets/site.css` for its appearance, and `assets/site.js` for navigation. The standalone example lives in `demo.html`, with its presentation in `assets/demo.css` and `assets/demo.js`.

## Research provenance

- The three images in `assets/study/` are unchanged copies from the supplied `Chessable Research_ Rating Your Moves.zip` export. The study text is a condensed, edited adaptation.
- `assets/demo-data.js` preserves the complete `DATA` object from the supplied `maia_inverse_bayes/visualizations/rating_your_moves_demo.html`: 52 half-moves, original boards, likelihoods, posteriors, and trajectories. The browser uses cached results; it does not run or query a model. The replay opens at White's seventh move to show a meaningful update; First returns to the opening. Finish plays 26… Nf2# on the board without another statistical update. Final position jumps directly to checkmate; Back restores the position before the move.
- Board-feature classifiers use 64 planes and game-average rating targets. The demo uses the Maia-3 5M human-move model to infer separate player distributions on a 600–2600 Lichess rating grid, with tempered evidence and a per-move trust cap.
- The displayed game is from Chess.com. Its header ratings and the inferred Lichess-scale estimates are not directly interchangeable. This is a one-game exploratory example, not a calibration or accuracy benchmark.
- The final section poses research questions about skill, style, and time pressure; it does not claim measured results for those directions.

Model source: [Maia Chess](https://www.maiachess.com/).

There is no build step or runtime dependency. Push changes to `main` to update the published site.
