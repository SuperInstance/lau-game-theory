# lau-game-theory

Game theory library for Rust: strategic interactions, equilibria, and mechanism design.

## Features

- **Normal form games**: Payoff matrices, dominant strategies, Nash equilibrium (pure and mixed), support enumeration, IESDS, Pareto efficiency
- **Zero-sum games**: Minimax theorem, value of the game, saddle points, 2x2 analytical solutions, general LP-style solving
- **Extensive form games**: Game trees, backward induction, subgame-perfect equilibria, chance nodes
- **Bayesian games**: Incomplete information, types, beliefs, Bayesian Nash equilibrium
- **Cooperative games**: Shapley value, core, nucleolus, superadditivity
- **Mechanism design**: Vickrey (second-price) auction, DSIC verification, VCG mechanism
- **Evolutionary game theory**: Replicator dynamics, evolutionary stable strategies (ESS)
- **Multi-agent strategy**: Agents as players, protocols as strategies, best-response dynamics, protocol adoption

## Usage

```rust
use lau_game_theory::NormalFormGame;
use nalgebra::DMatrix;

// Prisoner's Dilemma
let a = DMatrix::from_row_slice(2, 2, &[-1.0, -3.0, 0.0, -2.0]);
let b = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, -3.0, -2.0]);
let game = NormalFormGame::two_player(a, b);
let ne = game.pure_nash_equilibria(); // [(1, 1)] - both defect
```

## License

MIT
