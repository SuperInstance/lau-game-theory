//! Multi-agent strategy: agents as players, protocols as strategies.
//! Applications of game theory to multi-agent systems.

use crate::normal_form::NormalFormGame;
use crate::nash_equilibrium::{find_mixed_ne_2x2, MixedNashEquilibrium};
use serde::{Deserialize, Serialize};

/// An agent in a multi-agent system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    /// Agent identifier.
    pub id: String,
    /// Available strategies/protocols.
    pub strategies: Vec<String>,
    /// Utility function over outcomes (for simplicity, a payoff table).
    pub utility: Vec<f64>,
}

/// A multi-agent interaction scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentGame {
    /// Participating agents.
    pub agents: Vec<Agent>,
    /// The underlying normal-form game.
    pub game: NormalFormGame,
}

impl MultiAgentGame {
    /// Create a new multi-agent game.
    pub fn new(agents: Vec<Agent>, game: NormalFormGame) -> Self {
        Self { agents, game }
    }

    /// Find Nash equilibria (for 2x2 games).
    pub fn find_equilibria(&self) -> Vec<MixedNashEquilibrium> {
        find_mixed_ne_2x2(&self.game)
    }

    /// Get agent strategies.
    pub fn agent_strategies(&self, agent_idx: usize) -> &[String] {
        &self.agents[agent_idx].strategies
    }

    /// Recommend a strategy based on Nash equilibrium.
    /// Returns the mixed strategy for the first equilibrium found.
    pub fn recommend_strategy(&self, agent_idx: usize) -> Option<Vec<(String, f64)>> {
        let eqs = self.find_equilibria();
        if eqs.is_empty() {
            return None;
        }

        let eq = &eqs[0];
        let sigma = if agent_idx == 0 {
            &eq.sigma_r
        } else {
            &eq.sigma_c
        };

        let strategies = &self.agents[agent_idx].strategies;
        let result: Vec<(String, f64)> = strategies
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), sigma[i]))
            .collect();
        Some(result)
    }

    /// Simulate a round of play given strategy choices.
    pub fn play(&self, row_strategy: usize, col_strategy: usize) -> (f64, f64) {
        self.game.payoff(row_strategy, col_strategy)
    }
}

/// A protocol selection game: agents choose between communication protocols.
pub fn protocol_selection_game() -> MultiAgentGame {
    let agent_a = Agent {
        id: "Agent A".into(),
        strategies: vec!["TCP".into(), "UDP".into()],
        utility: vec![0.0; 4],
    };
    let agent_b = Agent {
        id: "Agent B".into(),
        strategies: vec!["TCP".into(), "UDP".into()],
        utility: vec![0.0; 4],
    };

    // Both prefer using the same protocol (coordination game)
    // TCP-TCP is best for both, UDP-UDP is okay
    let game = NormalFormGame::new(
        nalgebra::DMatrix::from_row_slice(2, 2, &[5.0, 0.0, 0.0, 3.0]),
        nalgebra::DMatrix::from_row_slice(2, 2, &[5.0, 0.0, 0.0, 3.0]),
    );

    MultiAgentGame::new(vec![agent_a, agent_b], game)
}

/// Resource allocation game: agents compete for limited resources.
pub fn resource_allocation_game(value: f64, cost_conflict: f64) -> MultiAgentGame {
    let agent_a = Agent {
        id: "Competitor A".into(),
        strategies: vec!["Compete".into(), "Yield".into()],
        utility: vec![0.0; 4],
    };
    let agent_b = Agent {
        id: "Competitor B".into(),
        strategies: vec!["Compete".into(), "Yield".into()],
        utility: vec![0.0; 4],
    };

    // Hawk-Dove style: compete vs yield
    let game = NormalFormGame::new(
        nalgebra::DMatrix::from_row_slice(2, 2, &[
            (value - cost_conflict) / 2.0, value,
            0.0, value / 2.0,
        ]),
        nalgebra::DMatrix::from_row_slice(2, 2, &[
            (value - cost_conflict) / 2.0, 0.0,
            value, value / 2.0,
        ]),
    );

    MultiAgentGame::new(vec![agent_a, agent_b], game)
}

/// Compute social welfare (sum of payoffs) at a given outcome.
pub fn social_welfare(game: &NormalFormGame, i: usize, j: usize) -> f64 {
    let (p1, p2) = game.payoff(i, j);
    p1 + p2
}

/// Find the outcome that maximizes social welfare.
pub fn max_social_welfare(game: &NormalFormGame) -> (usize, usize, f64) {
    let mut best = (0, 0, f64::NEG_INFINITY);
    for i in 0..game.n_row_strategies() {
        for j in 0..game.n_col_strategies() {
            let sw = social_welfare(game, i, j);
            if sw > best.2 {
                best = (i, j, sw);
            }
        }
    }
    best
}

/// Compute the price of anarchy: ratio of optimal social welfare to worst NE welfare.
pub fn price_of_anarchy(game: &NormalFormGame) -> f64 {
    let (_, _, opt) = max_social_welfare(game);
    let ne = game.pure_nash_equilibria();

    if ne.is_empty() || opt <= 0.0 {
        return f64::INFINITY;
    }

    let mut worst_ne_welfare = f64::INFINITY;
    for &(i, j) in &ne {
        let sw = social_welfare(game, i, j);
        if sw < worst_ne_welfare {
            worst_ne_welfare = sw;
        }
    }

    if worst_ne_welfare <= 0.0 {
        return f64::INFINITY;
    }

    opt / worst_ne_welfare
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normal_form::prisoners_dilemma;

    #[test]
    fn test_protocol_selection() {
        let mag = protocol_selection_game();
        let eqs = mag.find_equilibria();
        // Coordination game: 2 pure NE + 1 mixed
        assert_eq!(eqs.len(), 3);
    }

    #[test]
    fn test_recommend_strategy() {
        let mag = protocol_selection_game();
        let rec = mag.recommend_strategy(0);
        assert!(rec.is_some());
        let rec = rec.unwrap();
        // Should have probabilities for both strategies
        assert_eq!(rec.len(), 2);
    }

    #[test]
    fn test_play() {
        let mag = protocol_selection_game();
        let (p1, p2) = mag.play(0, 0); // TCP vs TCP
        assert!((p1 - 5.0).abs() < 1e-10);
        assert!((p2 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_social_welfare() {
        let g = prisoners_dilemma(5.0, 3.0, 1.0, 0.0);
        // (Cooperate, Cooperate): welfare = 3 + 3 = 6
        assert!((social_welfare(&g, 0, 0) - 6.0).abs() < 1e-10);
        // (Defect, Defect): welfare = 1 + 1 = 2
        assert!((social_welfare(&g, 1, 1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_social_welfare() {
        let g = prisoners_dilemma(5.0, 3.0, 1.0, 0.0);
        let (i, j, sw) = max_social_welfare(&g);
        assert_eq!((i, j), (0, 0)); // (C, C) maximizes welfare
        assert!((sw - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_price_of_anarchy_pd() {
        let g = prisoners_dilemma(5.0, 3.0, 1.0, 0.0);
        let poa = price_of_anarchy(&g);
        // Opt = 6, worst NE = 2 → PoA = 3
        assert!((poa - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_resource_allocation() {
        let mag = resource_allocation_game(10.0, 6.0);
        let eqs = mag.find_equilibria();
        assert!(!eqs.is_empty());
    }
}
