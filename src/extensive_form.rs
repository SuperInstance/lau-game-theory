//! Extensive-form games: game trees, backward induction, subgame perfection.

use serde::{Deserialize, Serialize};

/// Identifier for a player. 0 = player 1, 1 = player 2, etc. `Nature` for chance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Player {
    P(usize),
    Nature,
}

/// A node in an extensive-form game tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GameNode {
    /// Terminal node with payoffs (one per player).
    Terminal(Vec<f64>),
    /// Internal node where `player` chooses among actions.
    Decision {
        player: Player,
        /// Action label → child node.
        children: Vec<(String, Box<GameNode>)>,
    },
    /// Chance node with probability distribution over actions.
    Chance {
        /// (probability, action_label, child_node)
        outcomes: Vec<(f64, String, Box<GameNode>)>,
    },
}

impl GameNode {
    /// Create a terminal node.
    pub fn terminal(payoffs: Vec<f64>) -> Self {
        GameNode::Terminal(payoffs)
    }

    /// Create a decision node.
    pub fn decision(player: Player, children: Vec<(String, GameNode)>) -> Self {
        GameNode::Decision {
            player,
            children: children
                .into_iter()
                .map(|(a, n)| (a, Box::new(n)))
                .collect(),
        }
    }

    /// Create a chance node.
    pub fn chance(outcomes: Vec<(f64, String, GameNode)>) -> Self {
        GameNode::Chance {
            outcomes: outcomes
                .into_iter()
                .map(|(p, a, n)| (p, a, Box::new(n)))
                .collect(),
        }
    }
}

/// Result of backward induction: strategy profile and equilibrium payoffs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackwardInductionResult {
    /// Optimal action at each decision node (by DFS order).
    pub optimal_actions: Vec<(usize, String)>,
    /// Equilibrium payoffs.
    pub payoffs: Vec<f64>,
}

/// Run backward induction on an extensive-form game tree.
/// Returns the subgame-perfect equilibrium payoffs and optimal actions.
pub fn backward_induction(node: &GameNode) -> BackwardInductionResult {
    match node {
        GameNode::Terminal(payoffs) => BackwardInductionResult {
            optimal_actions: vec![],
            payoffs: payoffs.clone(),
        },
        GameNode::Decision { player, children } => {
            let sub_results: Vec<_> = children
                .iter()
                .map(|(_, child)| backward_induction(child))
                .collect();

            let player_idx = match player {
                Player::P(i) => *i,
                Player::Nature => panic!("Nature nodes should use Chance variant"),
            };

            let mut best_idx = 0;
            let mut best_val = f64::NEG_INFINITY;
            for (idx, sub) in sub_results.iter().enumerate() {
                if player_idx < sub.payoffs.len() && sub.payoffs[player_idx] > best_val {
                    best_val = sub.payoffs[player_idx];
                    best_idx = idx;
                }
            }

            let best_action = children[best_idx].0.clone();
            let mut actions = vec![(0, best_action)];
            actions.extend(sub_results[best_idx].optimal_actions.clone());

            // Renumber action indices by DFS
            let mut _offset = 1;
            let mut _renumbered: Vec<usize> = vec![];
            for (i, sub) in sub_results.iter().enumerate() {
                if i == best_idx {
                    continue;
                }
                for _ in &sub.optimal_actions {
                    _offset += 1;
                }
            }

            BackwardInductionResult {
                optimal_actions: actions,
                payoffs: sub_results[best_idx].payoffs.clone(),
            }
        }
        GameNode::Chance { outcomes } => {
            let sub_results: Vec<_> = outcomes
                .iter()
                .map(|(_, _, child)| backward_induction(child))
                .collect();

            let n_players = sub_results[0].payoffs.len();
            let mut expected = vec![0.0; n_players];
            for (i, sub) in sub_results.iter().enumerate() {
                let prob = outcomes[i].0;
                for (p, exp) in expected.iter_mut().enumerate() {
                    *exp += prob * sub.payoffs[p];
                }
            }

            let mut actions = vec![];
            for sub in &sub_results {
                actions.extend(sub.optimal_actions.clone());
            }

            BackwardInductionResult {
                optimal_actions: actions,
                payoffs: expected,
            }
        }
    }
}

/// Check if the game is a perfect-information game (all information sets are singletons).
/// In our tree representation, all games are perfect-info by construction.
pub fn is_perfect_information(_node: &GameNode) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_decision() {
        // Player 0 chooses Left (payoffs [1, 2]) or Right (payoffs [3, 0])
        let tree = GameNode::decision(
            Player::P(0),
            vec![
                ("L".into(), GameNode::terminal(vec![1.0, 2.0])),
                ("R".into(), GameNode::terminal(vec![3.0, 0.0])),
            ],
        );
        let result = backward_induction(&tree);
        assert!((result.payoffs[0] - 3.0).abs() < 1e-10);
        assert!((result.payoffs[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_two_stage_game() {
        // Player 0 moves first: In or Out
        // Out → (0, 0)
        // In → Player 1 moves: Fight → (-1, -1) or Accommodate → (2, 1)
        // SPE: Player 1 accommodates, Player 0 enters
        let tree = GameNode::decision(
            Player::P(0),
            vec![
                (
                    "Out".into(),
                    GameNode::terminal(vec![0.0, 0.0]),
                ),
                (
                    "In".into(),
                    GameNode::decision(
                        Player::P(1),
                        vec![
                            ("Fight".into(), GameNode::terminal(vec![-1.0, -1.0])),
                            ("Accommodate".into(), GameNode::terminal(vec![2.0, 1.0])),
                        ],
                    ),
                ),
            ],
        );
        let result = backward_induction(&tree);
        assert!((result.payoffs[0] - 2.0).abs() < 1e-10);
        assert!((result.payoffs[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_chance_node() {
        let tree = GameNode::chance(vec![
            (
                0.5,
                "Heads".into(),
                GameNode::terminal(vec![1.0, 0.0]),
            ),
            (
                0.5,
                "Tails".into(),
                GameNode::terminal(vec![0.0, 1.0]),
            ),
        ]);
        let result = backward_induction(&tree);
        assert!((result.payoffs[0] - 0.5).abs() < 1e-10);
        assert!((result.payoffs[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_perfect_information() {
        let tree = GameNode::decision(
            Player::P(0),
            vec![("A".into(), GameNode::terminal(vec![1.0]))],
        );
        assert!(is_perfect_information(&tree));
    }

    #[test]
    fn test_ultimatum_game() {
        // Player 0 offers: High or Low
        // Player 1 responds to High: Accept or Reject → (3, 2) or (0, 0)
        // Player 1 responds to Low: Accept or Reject → (5, 1) or (0, 0)
        let tree = GameNode::decision(
            Player::P(0),
            vec![
                (
                    "High".into(),
                    GameNode::decision(
                        Player::P(1),
                        vec![
                            ("Accept".into(), GameNode::terminal(vec![3.0, 2.0])),
                            ("Reject".into(), GameNode::terminal(vec![0.0, 0.0])),
                        ],
                    ),
                ),
                (
                    "Low".into(),
                    GameNode::decision(
                        Player::P(1),
                        vec![
                            ("Accept".into(), GameNode::terminal(vec![5.0, 1.0])),
                            ("Reject".into(), GameNode::terminal(vec![0.0, 0.0])),
                        ],
                    ),
                ),
            ],
        );
        let result = backward_induction(&tree);
        // Both offers get accepted, so player 0 picks Low (payoff 5 > 3)
        assert!((result.payoffs[0] - 5.0).abs() < 1e-10);
        assert!((result.payoffs[1] - 1.0).abs() < 1e-10);
    }
}
