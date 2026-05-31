//! Cooperative games: characteristic function, Shapley value, core, nucleolus.

use serde::{Deserialize, Serialize};

/// A cooperative (coalitional) game with transferable utility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CooperativeGame {
    /// Number of players.
    pub n_players: usize,
    /// Characteristic function: v(S) for each coalition S.
    /// Stored as a map from coalition (as sorted Vec of player indices) to value.
    pub v: Vec<f64>, // Indexed by coalition bitmask
}

impl CooperativeGame {
    /// Create a cooperative game from the characteristic function.
    /// `values` is indexed by coalition bitmask (bit i = player i in coalition).
    pub fn new(n_players: usize, values: Vec<f64>) -> Self {
        assert_eq!(values.len(), 1 << n_players, "Must have 2^n values");
        Self { n_players, v: values }
    }

    /// Get the value of coalition represented by bitmask.
    pub fn value_of(&self, coalition: u64) -> f64 {
        self.v[coalition as usize]
    }

    /// Grand coalition value.
    pub fn grand_coalition_value(&self) -> f64 {
        self.v[(1 << self.n_players) - 1]
    }

    /// Compute the Shapley value for all players.
    pub fn shapley_value(&self) -> Vec<f64> {
        let n = self.n_players;
        let mut phi = vec![0.0; n];

        for (player, phi_entry) in phi.iter_mut().enumerate() {
            let mut total = 0.0;
            let all_players: u64 = ((1u64 << n) - 1) & !(1u64 << player);

            // Enumerate all subsets S of players \\ {player} using bitmask
            for subset in 0u64..=(all_players) {
                // Only consider subsets of all_players
                if subset & !all_players != 0 {
                    continue;
                }
                let s_size = subset.count_ones() as usize;
                let with_player = subset | (1u64 << player);
                let marginal = self.v[with_player as usize] - self.v[subset as usize];

                // Weight: |S|! * (n - |S| - 1)! / n!
                let weight = factorial(s_size) * factorial(n - s_size - 1)
                    / factorial(n);

                total += weight * marginal;
            }

            *phi_entry = total;
        }

        phi
    }

    /// Check if an allocation is in the core.
    /// An allocation x is in the core iff:
    /// 1. sum(x) = v(N) (efficiency)
    /// 2. For all coalitions S: sum_{i in S}(x_i) >= v(S) (coalitional rationality)
    pub fn is_in_core(&self, allocation: &[f64]) -> bool {
        let n = self.n_players;
        assert_eq!(allocation.len(), n);

        // Efficiency
        let total: f64 = allocation.iter().sum();
        if (total - self.grand_coalition_value()).abs() > 1e-10 {
            return false;
        }

        // Coalitional rationality
        for coalition in 1u64..(1u64 << n) {
            let coalition_value = self.v[coalition as usize];
            let allocation_sum = sum_allocation(allocation, coalition);
            if allocation_sum < coalition_value - 1e-10 {
                return false;
            }
        }
        true
    }

    /// Find the core (all extreme points) — brute force for small games.
    /// Returns allocations that are in the core, tested via sampling.
    pub fn core_vertices(&self) -> Vec<Vec<f64>> {
        let n = self.n_players;
        if n > 6 {
            return vec![]; // Too large
        }

        let grand = self.grand_coalition_value();
        let mut vertices = vec![];

        // For n=2: enumerate allocations on a grid
        if n == 2 {
            for i in 0..=100 {
                let x0 = grand * (i as f64) / 100.0;
                let x1 = grand - x0;
                let alloc = vec![x0, x1];
                if self.is_in_core(&alloc) {
                    vertices.push(alloc);
                }
            }
        } else if n == 3 {
            // Sample on a grid
            for i in 0..=20 {
                for j in 0..=(20 - i) {
                    let x0 = grand * (i as f64) / 20.0;
                    let x1 = grand * (j as f64) / 20.0;
                    let x2 = grand - x0 - x1;
                    let alloc = vec![x0, x1, x2];
                    if self.is_in_core(&alloc) {
                        vertices.push(alloc);
                    }
                }
            }
        }

        vertices
    }

    /// Compute the nucleolus (approximate) using a simplified approach.
    /// Returns the nucleolus allocation.
    pub fn nucleolus(&self) -> Vec<f64> {
        let n = self.n_players;
        let phi = self.shapley_value();

        // Start with Shapley value and try to improve
        let mut best = phi.clone();
        let mut best_min_excess = self.min_excess(&best);

        // Simple iterative improvement
        for _ in 0..1000 {
            let mut improved = false;
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let delta = 0.001 * self.grand_coalition_value();
                    let mut trial = best.clone();
                    trial[i] += delta;
                    trial[j] -= delta;
                    // Keep non-negative
                    if trial[j] < 0.0 {
                        continue;
                    }
                    let trial_excess = self.min_excess(&trial);
                    if trial_excess > best_min_excess + 1e-12 {
                        best = trial;
                        best_min_excess = trial_excess;
                        improved = true;
                    }
                }
            }
            if !improved {
                break;
            }
        }

        best
    }

    fn min_excess(&self, allocation: &[f64]) -> f64 {
        let n = self.n_players;
        let mut min_excess = f64::INFINITY;
        for coalition in 1u64..((1u64 << n) - 1) {
            let coalition_value = self.v[coalition as usize];
            let allocation_sum = sum_allocation(allocation, coalition);
            let excess = allocation_sum - coalition_value;
            if excess < min_excess {
                min_excess = excess;
            }
        }
        min_excess
    }
}

fn factorial(n: usize) -> f64 {
    let mut result = 1.0;
    for i in 2..=n {
        result *= i as f64;
    }
    result
}

fn sum_allocation(allocation: &[f64], coalition: u64) -> f64 {
    let mut sum = 0.0;
    let mut c = coalition;
    let mut idx = 0;
    while c > 0 {
        if c & 1 != 0 {
            sum += allocation[idx];
        }
        c >>= 1;
        idx += 1;
    }
    sum
}


/// Create a simple voting game: v(S) = 1 if |S| > n/2, else 0.
pub fn simple_voting_game(n_players: usize) -> CooperativeGame {
    let quota = n_players / 2 + 1;
    let mut values = vec![0.0; 1 << n_players];
    for coalition in 0u64..(1u64 << n_players) {
        if coalition.count_ones() as usize >= quota {
            values[coalition as usize] = 1.0;
        }
    }
    CooperativeGame::new(n_players, values)
}

/// Create a glove game: half players have left gloves, half have right gloves.
/// v(S) = min(left_in_S, right_in_S).
pub fn glove_game(n_left: usize, n_right: usize) -> CooperativeGame {
    let n = n_left + n_right;
    let mut values = vec![0.0; 1 << n];
    for coalition in 0u64..(1u64 << n) {
        let mut left = 0;
        let mut right = 0;
        for i in 0..n {
            if coalition & (1u64 << i) != 0 {
                if i < n_left {
                    left += 1;
                } else {
                    right += 1;
                }
            }
        }
        values[coalition as usize] = left.min(right) as f64;
    }
    CooperativeGame::new(n, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shapley_unanimity() {
        // 3 players, v(S) = 1 only for grand coalition
        let mut values = vec![0.0; 8];
        values[7] = 1.0;
        let game = CooperativeGame::new(3, values);
        let phi = game.shapley_value();
        // Symmetric → all equal
        assert!((phi[0] - 1.0 / 3.0).abs() < 1e-10);
        assert!((phi[1] - 1.0 / 3.0).abs() < 1e-10);
        assert!((phi[2] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_shapley_dictator() {
        // Player 0 is a dictator: v(S) = 1 iff 0 ∈ S
        let n = 3;
        let mut values = vec![0.0; 8];
        for c in 1u64..8 {
            if c & 1 != 0 {
                values[c as usize] = 1.0;
            }
        }
        let game = CooperativeGame::new(n, values);
        let phi = game.shapley_value();
        assert!((phi[0] - 1.0).abs() < 1e-10);
        assert!((phi[1]).abs() < 1e-10);
        assert!((phi[2]).abs() < 1e-10);
    }

    #[test]
    fn test_core_unanimity() {
        // 3 players, v(S) = 1 only for grand coalition
        let mut values = vec![0.0; 8];
        values[7] = 1.0;
        let game = CooperativeGame::new(3, values);
        // (1/3, 1/3, 1/3) should be in core
        assert!(game.is_in_core(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]));
        // (1, 0, 0) IS in core for unanimity (only grand coalition has value)
        assert!(game.is_in_core(&[1.0, 0.0, 0.0]));
        // (0.4, 0.3, 0.2) should be in core
        assert!(game.is_in_core(&[0.4, 0.3, 0.3]));
    }

    #[test]
    fn test_glove_game_shapley() {
        let game = glove_game(1, 1);
        let phi = game.shapley_value();
        // Symmetric: each gets 0.5
        assert!((phi[0] - 0.5).abs() < 1e-10);
        assert!((phi[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_voting_game() {
        let game = simple_voting_game(3);
        let phi = game.shapley_value();
        // All symmetric
        for i in 0..3 {
            assert!((phi[i] - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_shapley_efficiency() {
        let mut values = vec![0.0; 8];
        values[7] = 10.0;
        values[3] = 2.0;
        values[5] = 3.0;
        let game = CooperativeGame::new(3, values);
        let phi = game.shapley_value();
        let sum: f64 = phi.iter().sum();
        assert!((sum - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_nucleolus_efficiency() {
        let mut values = vec![0.0; 8];
        values[7] = 6.0;
        let game = CooperativeGame::new(3, values);
        let nuc = game.nucleolus();
        let sum: f64 = nuc.iter().sum();
        assert!((sum - 6.0).abs() < 1e-6);
    }
}
