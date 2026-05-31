//! Bayesian games: incomplete information, types, beliefs.

use serde::{Deserialize, Serialize};

/// A type profile for players in a Bayesian game.
pub type TypeProfile = Vec<usize>;

/// A Bayesian (incomplete-information) game.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianGame {
    /// Number of players.
    pub n_players: usize,
    /// Number of possible types per player.
    pub n_types: Vec<usize>,
    /// Number of actions per player.
    pub n_actions: Vec<usize>,
    /// Common prior over type profiles: P(type_1, type_2, ..., type_n).
    /// Indexed by flattening the type profile.
    pub prior: Vec<f64>,
    /// Payoff function: payoffs[player][type_profile_flat][action_profile_flat].
    pub payoffs: Vec<Vec<Vec<f64>>>,
}

impl BayesianGame {
    /// Create a simple 2-player Bayesian game with independent types.
    #[allow(clippy::too_many_arguments)]
    pub fn new_independent_2player(
        n_types_p1: usize,
        n_types_p2: usize,
        n_actions_p1: usize,
        n_actions_p2: usize,
        type_probs_p1: Vec<f64>,
        type_probs_p2: Vec<f64>,
        payoffs_p1: Vec<Vec<Vec<f64>>>, // [t1][t2][(a1*n_actions_p2 + a2)]
        payoffs_p2: Vec<Vec<Vec<f64>>>,
    ) -> Self {
        let mut prior = Vec::with_capacity(n_types_p1 * n_types_p2);
        for (_t1, tp1) in type_probs_p1.iter().enumerate().take(n_types_p1) {
            for (_t2, tp2) in type_probs_p2.iter().enumerate().take(n_types_p2) {
                prior.push(*tp1 * *tp2);
            }
        }

        let mut all_payoffs_p1 = vec![];
        let mut all_payoffs_p2 = vec![];
        for t1 in 0..n_types_p1 {
            for t2 in 0..n_types_p2 {
                all_payoffs_p1.push(payoffs_p1[t1][t2].clone());
                all_payoffs_p2.push(payoffs_p2[t1][t2].clone());
            }
        }

        Self {
            n_players: 2,
            n_types: vec![n_types_p1, n_types_p2],
            n_actions: vec![n_actions_p1, n_actions_p2],
            prior,
            payoffs: vec![all_payoffs_p1, all_payoffs_p2],
        }
    }

    /// Compute the posterior belief of player i about opponent types,
    /// given player i's type.
    pub fn posterior_belief(&self, player: usize, own_type: usize) -> Vec<f64> {
        let other = 1 - player;
        let n_other_types = self.n_types[other];

        let mut numerators = vec![0.0; n_other_types];
        let mut total = 0.0;

        // Iterate over all type profiles
        let mut idx = 0;
        for t1 in 0..self.n_types[0] {
            for t2 in 0..self.n_types[1] {
                let type_matches = if player == 0 { t1 == own_type } else { t2 == own_type };
                if type_matches {
                    let other_type = if player == 0 { t2 } else { t1 };
                    numerators[other_type] += self.prior[idx];
                    total += self.prior[idx];
                }
                idx += 1;
            }
        }

        if total > 0.0 {
            for n in numerators.iter_mut() {
                *n /= total;
            }
        }
        numerators
    }

    /// Compute expected payoff for a player given a strategy profile.
    /// `strategies[player][type]` = chosen action index.
    pub fn expected_payoff(&self, player: usize, strategies: &[Vec<usize>]) -> f64 {
        let mut expected = 0.0;
        let mut idx = 0;
        for t1 in 0..self.n_types[0] {
            for t2 in 0..self.n_types[1] {
                let a1 = strategies[0][t1];
                let a2 = strategies[1][t2];
                let action_idx = a1 * self.n_actions[1] + a2;
                expected += self.prior[idx] * self.payoffs[player][idx][action_idx];
                idx += 1;
            }
        }
        expected
    }

    /// Find pure-strategy Bayesian Nash equilibria by brute force (small games only).
    pub fn find_pure_bne(&self) -> Vec<Vec<Vec<usize>>> {
        let mut equilibria = vec![];

        // Enumerate all strategy profiles
        let mut type_action_sizes: Vec<Vec<usize>> = vec![];
        for p in 0..self.n_players {
            type_action_sizes.push(vec![self.n_actions[p]; self.n_types[p]]);
        }
        // For 2 players with small type/action spaces
        if self.n_players != 2 {
            return equilibria;
        }

        // Generate all strategy combos for player 0
        let p0_strats = self.enumerate_strategies(0);
        let p1_strats = self.enumerate_strategies(1);

        for s0 in &p0_strats {
            for s1 in &p1_strats {
                let strategies = vec![s0.clone(), s1.clone()];

                // Check if any player wants to deviate
                let mut is_ne = true;
                for p in 0..2 {
                    let current_payoff = self.expected_payoff(p, &strategies);
                    // Check all deviations
                    let dev_strats = self.enumerate_strategies(p);
                    for dev in &dev_strats {
                        let mut test_strats = strategies.clone();
                        test_strats[p] = dev.clone();
                        let dev_payoff = self.expected_payoff(p, &test_strats);
                        if dev_payoff > current_payoff + 1e-10 {
                            is_ne = false;
                            break;
                        }
                    }
                    if !is_ne {
                        break;
                    }
                }

                if is_ne {
                    equilibria.push(strategies);
                }
            }
        }

        equilibria
    }

    fn enumerate_strategies(&self, player: usize) -> Vec<Vec<usize>> {
        let n_types = self.n_types[player];
        let n_actions = self.n_actions[player];
        let mut result = vec![vec![0usize; n_types]];
        for t in 0..n_types {
            let mut new_result = vec![];
            for existing in &result {
                for a in 0..n_actions {
                    let mut s = existing.clone();
                    s[t] = a;
                    new_result.push(s);
                }
            }
            result = new_result;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_bayesian_game() {
        // 2 players, each 2 types, 2 actions
        // Type probs: P1(t=0)=0.5, P1(t=1)=0.5; same for P2
        let bg = BayesianGame::new_independent_2player(
            2, 2, 2, 2,
            vec![0.5, 0.5],
            vec![0.5, 0.5],
            // P1 payoffs: [t1][t2][(a1*2+a2)]
            vec![
                vec![
                    vec![2.0, 0.0, 0.0, 1.0], // t1=0, t2=0
                    vec![2.0, 0.0, 0.0, 1.0], // t1=0, t2=1
                ],
                vec![
                    vec![0.0, 1.0, 2.0, 0.0], // t1=1, t2=0
                    vec![0.0, 1.0, 2.0, 0.0], // t1=1, t2=1
                ],
            ],
            // P2 payoffs
            vec![
                vec![
                    vec![1.0, 2.0, 0.0, 0.0],
                    vec![1.0, 2.0, 0.0, 0.0],
                ],
                vec![
                    vec![1.0, 2.0, 0.0, 0.0],
                    vec![1.0, 2.0, 0.0, 0.0],
                ],
            ],
        );
        let bne = bg.find_pure_bne();
        assert!(!bne.is_empty());
    }

    #[test]
    fn test_posterior_belief() {
        let bg = BayesianGame::new_independent_2player(
            2, 2, 2, 2,
            vec![0.6, 0.4],
            vec![0.3, 0.7],
            // dummy payoffs: [t1][t2][action_idx]
            vec![
                vec![vec![0.0; 4]; 2],
                vec![vec![0.0; 4]; 2],
            ],
            vec![
                vec![vec![0.0; 4]; 2],
                vec![vec![0.0; 4]; 2],
            ],
        );
        // P(player 2 type=0 | player 1 type=0) should be 0.3
        let belief = bg.posterior_belief(0, 0);
        assert!((belief[0] - 0.3).abs() < 1e-10);
        assert!((belief[1] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_expected_payoff() {
        let bg = BayesianGame::new_independent_2player(
            1, 1, 2, 2,
            vec![1.0],
            vec![1.0],
            vec![vec![vec![3.0, 0.0, 0.0, 1.0]]],
            vec![vec![vec![1.0, 2.0, 0.0, 0.0]]],
        );
        // Both always type 0, strategies: [0] and [0]
        let strategies = vec![vec![0], vec![0]];
        let ep = bg.expected_payoff(0, &strategies);
        assert!((ep - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_prior_sums_to_one() {
        let bg = BayesianGame::new_independent_2player(
            2, 2, 2, 2,
            vec![0.5, 0.5],
            vec![0.5, 0.5],
            vec![vec![vec![0.0; 4]; 2], vec![vec![0.0; 4]; 2]],
            vec![vec![vec![0.0; 4]; 2], vec![vec![0.0; 4]; 2]],
        );
        let sum: f64 = bg.prior.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
