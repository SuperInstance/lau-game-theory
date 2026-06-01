# lau-game-theory

A Rust library for **game theory** — normal-form games, Nash equilibrium, zero-sum games, extensive-form games, Bayesian games, cooperative games, mechanism design, evolutionary dynamics, and multi-agent applications.

## What This Does

This crate covers the major branches of game theory with working implementations:

- **Normal-form games** — Payoff matrices, best-response correspondences, pure Nash equilibria, strict dominance, iterated elimination of dominated strategies (IESDS), expected utility under mixed strategies
- **Nash equilibrium** — Analytical 2×2 solver, support enumeration for 2×N games, mixed-strategy equilibria
- **Zero-sum games** — Minimax theorem, saddle points, maximin/minimax values, analytical 2×2 solver, dominant strategy detection
- **Extensive-form games** — Game trees (decision nodes, chance nodes, terminal nodes), backward induction for subgame-perfect equilibrium
- **Bayesian games** — Incomplete information with types, common priors, posterior beliefs, brute-force Bayesian Nash equilibrium
- **Cooperative games** — Characteristic functions, Shapley value, core membership, nucleolus (approximate), glove games, voting games
- **Mechanism design** — Vickrey (second-price) auction, first-price auction, all-pay auction, VCG mechanism, incentive compatibility verification
- **Evolutionary game theory** — Replicator dynamics, evolutionarily stable strategies (ESS), Hawk-Dove, coordination games
- **Multi-agent applications** — Protocol selection games, resource allocation, social welfare, price of anarchy

## Key Idea

Game theory studies strategic interaction: what rational agents do when their outcomes depend on *each other's* choices. This library treats games as structured data — payoff matrices, game trees, characteristic functions — and provides algorithms for computing equilibria (Nash, subgame-perfect, Bayesian, evolutionary).

The core computational pattern is **equilibrium finding**: a Nash equilibrium is a strategy profile where no player benefits from unilateral deviation. For 2×2 games this is solvable analytically; for larger games, support enumeration checks all candidate support sets. The library also handles cooperative solution concepts (Shapley value, core) and mechanism design (ensuring incentive compatibility via Vickrey/VCG payments).

## Install

```toml
[dependencies]
lau-game-theory = "0.1"
```

Requires **Rust 2021 edition**.

### Dependencies

| Crate | Purpose |
|-------|---------|
| `nalgebra` | Matrices and vectors for payoff representation |
| `serde` | Serialization of games, equilibria, outcomes |

## Quick Start

### Normal-form game

```rust
use lau_game_theory::normal_form::{NormalFormGame, prisoners_dilemma};
use nalgebra::dmatrix;

// Prisoner's Dilemma
let g = prisoners_dilemma(5.0, 3.0, 1.0, 0.0);
//                Temptation, Reward, Punishment, Sucker

// Find pure Nash equilibria
let ne = g.pure_nash_equilibria();
assert_eq!(ne, vec![(1, 1)]); // (Defect, Defect)

// Check dominance
assert!(g.strictly_dominates_row(1, 0)); // Defect dominates Cooperate
```

### Mixed-strategy Nash equilibrium

```rust
use lau_game_theory::nash_equilibrium::find_mixed_ne_2x2;
use lau_game_theory::normal_form::{matching_pennies, battle_of_the_sexes};

// Matching Pennies: no pure NE, one mixed NE
let g = matching_pennies();
let ne = find_mixed_ne_2x2(&g);
let mixed = ne.iter().find(|eq| eq.sigma_r[0] > 0.01 && eq.sigma_r[0] < 0.99).unwrap();
assert!((mixed.sigma_r[0] - 0.5).abs() < 1e-10); // 50-50

// Battle of the Sexes: 2 pure + 1 mixed = 3 NE
let g = battle_of_the_sexes();
assert_eq!(find_mixed_ne_2x2(&g).len(), 3);
```

### Zero-sum game

```rust
use lau_game_theory::zero_sum::ZeroSumGame;
use nalgebra::dmatrix;

let g = ZeroSumGame::new(dmatrix![1.0, -1.0; -1.0, 1.0]); // Matching Pennies

// No saddle point
assert!(g.saddle_points().is_empty());
assert!(g.value_pure().is_none());

// Solve analytically: ((0.5, 0.5), (0.5, 0.5), 0.0)
let ((p, _), (q, _), val) = g.solve_2x2().unwrap();
assert!((p - 0.5).abs() < 1e-10);
assert!(val.abs() < 1e-10);
```

### Extensive-form game

```rust
use lau_game_theory::extensive_form::{GameNode, Player, backward_induction};

// Market entry game: Player 0 chooses In/Out, Player 1 responds
let tree = GameNode::decision(
    Player::P(0),
    vec![
        ("Out".into(), GameNode::terminal(vec![0.0, 0.0])),
        ("In".into(), GameNode::decision(
            Player::P(1),
            vec![
                ("Fight".into(), GameNode::terminal(vec![-1.0, -1.0])),
                ("Accommodate".into(), GameNode::terminal(vec![2.0, 1.0])),
            ],
        )),
    ],
);

let result = backward_induction(&tree);
assert!((result.payoffs[0] - 2.0).abs() < 1e-10); // Player 0 enters
assert!((result.payoffs[1] - 1.0).abs() < 1e-10); // Player 1 accommodates
```

### Cooperative game and Shapley value

```rust
use lau_game_theory::cooperative::CooperativeGame;

// 3-player game: only the grand coalition has value 10
let mut values = vec![0.0; 8];
values[7] = 10.0;
values[3] = 2.0;
values[5] = 3.0;

let game = CooperativeGame::new(3, values);
let shapley = game.shapley_value();

// Efficiency: Shapley values sum to v(N) = 10
let sum: f64 = shapley.iter().sum();
assert!((sum - 10.0).abs() < 1e-10);

// Check core membership
assert!(game.is_in_core(&[4.0, 3.0, 3.0]));
```

### Mechanism design

```rust
use lau_game_theory::mechanism::{vickrey_auction, DirectMechanism};

// Second-price auction
let outcome = vickrey_auction(&[10.0, 20.0, 15.0]);
assert_eq!(outcome.winner, Some(1)); // Highest bidder
assert!((outcome.price - 15.0).abs() < 1e-10); // Pays second price

// VCG mechanism
let mech = DirectMechanism::new(3, 1);
let result = mech.vcg_single_item(&[10.0, 30.0, 20.0]);
assert!(result[1].0);   // Player 1 wins
assert!((result[1].1 - 20.0).abs() < 1e-10); // Pays externality = second price
```

### Evolutionary game theory

```rust
use lau_game_theory::evolutionary::{hawk_dove_game, EvolutionaryGame};
use nalgebra::dvector;

let game = hawk_dove_game(4.0, 6.0); // V=4, C=6

// ESS: mix with 2/3 Hawk, 1/3 Dove
let x_star = dvector![2.0/3.0, 1.0/3.0];
assert!(game.is_ess(&x_star));

// Pure strategies are NOT ESS
assert!(!game.is_ess(&dvector![1.0, 0.0]));

// Replicator dynamics converge to ESS
let trajectory = game.replicator_dynamics(&dvector![0.5, 0.5], 0.1, 200);
```

### Multi-agent: price of anarchy

```rust
use lau_game_theory::multi_agent::{social_welfare, price_of_anarchy};
use lau_game_theory::normal_form::prisoners_dilemma;

let g = prisoners_dilemma(5.0, 3.0, 1.0, 0.0);

// Optimal welfare: (Cooperate, Cooperate) = 6
assert!((social_welfare(&g, 0, 0) - 6.0).abs() < 1e-10);

// Nash welfare: (Defect, Defect) = 2
// Price of anarchy = 6/2 = 3
assert!((price_of_anarchy(&g) - 3.0).abs() < 1e-10);
```

## API Reference

### `normal_form` — Normal-Form Games

| Item | Description |
|------|-------------|
| `NormalFormGame` | Two-player game with `row_payoffs` and `col_payoffs` matrices |
| `best_response_row(j)` / `best_response_col(i)` | Best-response sets |
| `pure_nash_equilibria()` | All pure NE as `(row, col)` pairs |
| `strictly_dominates_row(i, k)` | Strict dominance check |
| `iesds_row()` | Iterated elimination of strictly dominated strategies |
| `expected_payoff_row(sigma_r, sigma_c)` | Expected utility under mixed strategies |

**Named games**: `prisoners_dilemma(t, r, p, s)`, `matching_pennies()`, `battle_of_the_sexes()`, `coordination_game()`.

### `nash_equilibrium` — Nash Equilibrium

| Function | Description |
|----------|-------------|
| `find_mixed_ne_2x2(game)` | All NE (pure + mixed) for 2×2 games |
| `support_enumeration(game)` | Support enumeration for 2×N and Nx2 games |

Returns `Vec<MixedNashEquilibrium>` with `sigma_r`, `sigma_c`, and `payoffs`.

### `zero_sum` — Zero-Sum Games

| Method | Description |
|--------|-------------|
| `saddle_points()` | Pure equilibria `(i, j, value)` |
| `maximin_value()` / `minimax_value()` | Lower/upper value of the game |
| `value_pure()` | Game value if saddle point exists |
| `solve_2x2()` | Analytical mixed-strategy solution |
| `dominant_row()` | Dominant strategy for player 1 |

### `extensive_form` — Game Trees

| Item | Description |
|------|-------------|
| `GameNode` | `Terminal(payoffs)`, `Decision { player, children }`, `Chance { outcomes }` |
| `Player` | `P(usize)` or `Nature` |
| `backward_induction(node)` | Subgame-perfect equilibrium via backward induction |
| `BackwardInductionResult` | `optimal_actions` and equilibrium `payoffs` |

### `bayesian` — Bayesian Games

| Method | Description |
|--------|-------------|
| `new_independent_2player(...)` | Create game with independent type distributions |
| `posterior_belief(player, type)` | P(opponent types \| own type) |
| `expected_payoff(player, strategies)` | Ex-ante expected utility |
| `find_pure_bne()` | Brute-force pure Bayesian NE |

### `cooperative` — Cooperative Games

| Method | Description |
|--------|-------------|
| `shapley_value()` | Fair division: `φᵢ = Σ_S [|S|!(n-|S|-1)!/n!] · [v(S∪{i}) - v(S)]` |
| `is_in_core(allocation)` | Efficiency + coalitional rationality |
| `core_vertices()` | Sample extreme points of the core |
| `nucleolus()` | Approximate lexicographic maximin excess allocation |
| `grand_coalition_value()` | `v(N)` |

**Named games**: `simple_voting_game(n)`, `glove_game(n_left, n_right)`.

### `mechanism` — Mechanism Design

| Function | Description |
|----------|-------------|
| `vickrey_auction(bids)` | Second-price sealed-bid auction |
| `first_price_auction(bids)` | First-price sealed-bid auction |
| `all_pay_auction(bids)` | All-pay auction |
| `DirectMechanism::vcg_single_item(values)` | VCG mechanism |
| `check_incentive_compatibility(...)` | Verify truth-telling is dominant |

### `evolutionary` — Evolutionary Game Theory

| Method | Description |
|--------|-------------|
| `replicator_step(x, dt)` | One step of replicator dynamics: `ẋᵢ = xᵢ(fᵢ(x) - f̄(x))` |
| `replicator_dynamics(x₀, dt, steps)` | Full trajectory |
| `is_ess(x*)` | Check evolutionarily stable strategy |
| `fitness(x)` | Payoff of each pure strategy against population `x` |
| `average_payoff(x)` | Mean population fitness `xᵀAx` |

**Named games**: `hawk_dove_game(v, c)`, `evolutionary_coordination(a, b)`.

### `multi_agent` — Multi-Agent Applications

| Function | Description |
|----------|-------------|
| `social_welfare(game, i, j)` | Sum of payoffs at outcome |
| `max_social_welfare(game)` | Socially optimal outcome |
| `price_of_anarchy(game)` | `opt_welfare / worst_NE_welfare` |
| `MultiAgentGame` | Wraps agents + game with `find_equilibria()`, `recommend_strategy()` |
| `protocol_selection_game()` | Coordination game for protocol choice |
| `resource_allocation_game(v, c)` | Hawk-Dove style competition |

## How It Works

### Nash equilibrium finding

For 2×2 games, the mixed-strategy NE is found analytically. Each player must be *indifferent* between their strategies given the opponent's mix:

```
P1 indifference:  p·A[0,0] + (1-p)·A[1,0] = p·A[0,1] + (1-p)·A[1,1]
P2 indifference:  q·B[0,0] + (1-q)·B[0,1] = q·B[1,0] + (1-q)·B[1,1]
```

Solving two linear equations gives the mixing probabilities.

### Backward induction

Starting from terminal nodes, each decision node is replaced with the outcome of the acting player's best choice. Chance nodes compute expected payoffs over their probability distribution. This yields the subgame-perfect equilibrium.

### Shapley value

For player `i`, the Shapley value averages the marginal contribution over all possible orderings of the other players:

```
φᵢ = Σ_{S ⊆ N\{i}} [|S|! · (n-|S|-1)! / n!] · [v(S ∪ {i}) - v(S)]
```

This is the unique allocation satisfying efficiency, symmetry, linearity, and the null player axiom.

### Replicator dynamics

Population state `x` evolves as:

```
ẋᵢ = xᵢ · (fᵢ(x) - f̄(x))
```

where `fᵢ(x) = (Ax)ᵢ` is the fitness of strategy `i` and `f̄(x) = xᵀAx` is the average fitness. Strategies with above-average fitness grow; those below shrink.

### Vickrey auction incentive compatibility

In a second-price auction, bidding your true value is a *dominant strategy*. If you win, you pay the second-highest bid regardless of your own bid. Overbidding risks paying more than your value; underbidding risks losing when you would have profited. The library verifies this formally by checking all possible misreports.

## The Math

**Nash's theorem** (1950): every finite game has at least one mixed-strategy Nash equilibrium. For 2-player games, these are found at points where both players are indifferent between all strategies in their support.

**Minimax theorem** (von Neumann, 1928): for two-player zero-sum games:

```
max_σ₁ min_σ₂ u(σ₁, σ₂) = min_σ₂ max_σ₁ u(σ₁, σ₂) = value of the game
```

**Shapley value axioms**: efficiency (`Σφᵢ = v(N)`), symmetry (identical players get same value), linearity (Shapley of sum = sum of Shapleys), null player (zero marginal contribution → zero value).

**ESS criterion** (Maynard Smith, 1973): strategy `x*` is an ESS if for all invaders `y ≠ x*`:

1. `u(x*, x*) > u(y, x*)`, or
2. `u(x*, x*) = u(y, x*)` and `u(x*, y) > u(y, y)`

**Price of anarchy**: ratio of the socially optimal welfare to the welfare at the worst Nash equilibrium. Measures the efficiency loss from selfish behavior. For Prisoner's Dilemma: `PoA = (R+R) / (P+P)`.

## License

MIT
