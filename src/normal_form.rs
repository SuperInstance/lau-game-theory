//! Normal-form (strategic-form) games: payoff matrices, dominant strategies,
//! best-response correspondences, and pure-strategy Nash equilibrium.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// A two-player normal-form game defined by payoff matrices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalFormGame {
    /// Row player (player 1) payoff matrix. Shape: (rows, cols).
    pub row_payoffs: DMatrix<f64>,
    /// Column player (player 2) payoff matrix. Shape: (rows, cols).
    pub col_payoffs: DMatrix<f64>,
}

impl NormalFormGame {
    /// Create a new normal-form game with the given payoff matrices.
    pub fn new(row_payoffs: DMatrix<f64>, col_payoffs: DMatrix<f64>) -> Self {
        assert_eq!(
            row_payoffs.shape(),
            col_payoffs.shape(),
            "Payoff matrices must have the same shape"
        );
        Self {
            row_payoffs,
            col_payoffs,
        }
    }

    /// Number of strategies for the row player.
    pub fn n_row_strategies(&self) -> usize {
        self.row_payoffs.nrows()
    }

    /// Number of strategies for the column player.
    pub fn n_col_strategies(&self) -> usize {
        self.row_payoffs.ncols()
    }

    /// Return the best-response set of row strategies against column strategy `j`.
    pub fn best_response_row(&self, j: usize) -> Vec<usize> {
        let mut best_val = f64::NEG_INFINITY;
        let mut best = vec![];
        for i in 0..self.n_row_strategies() {
            let v = self.row_payoffs[(i, j)];
            if v > best_val {
                best_val = v;
                best = vec![i];
            } else if (v - best_val).abs() < 1e-12 {
                best.push(i);
            }
        }
        best
    }

    /// Return the best-response set of column strategies against row strategy `i`.
    pub fn best_response_col(&self, i: usize) -> Vec<usize> {
        let mut best_val = f64::NEG_INFINITY;
        let mut best = vec![];
        for j in 0..self.n_col_strategies() {
            let v = self.col_payoffs[(i, j)];
            if v > best_val {
                best_val = v;
                best = vec![j];
            } else if (v - best_val).abs() < 1e-12 {
                best.push(j);
            }
        }
        best
    }

    /// Find all pure-strategy Nash equilibria as (row, col) pairs.
    pub fn pure_nash_equilibria(&self) -> Vec<(usize, usize)> {
        let mut equilibria = vec![];
        for i in 0..self.n_row_strategies() {
            for j in 0..self.n_col_strategies() {
                let br_row = self.best_response_row(j);
                let br_col = self.best_response_col(i);
                if br_row.contains(&i) && br_col.contains(&j) {
                    equilibria.push((i, j));
                }
            }
        }
        equilibria
    }

    /// Check if row strategy `i` strictly dominates row strategy `k`.
    pub fn strictly_dominates_row(&self, i: usize, k: usize) -> bool {
        for j in 0..self.n_col_strategies() {
            if self.row_payoffs[(i, j)] <= self.row_payoffs[(k, j)] {
                return false;
            }
        }
        true
    }

    /// Check if col strategy `j` strictly dominates col strategy `l`.
    pub fn strictly_dominates_col(&self, j: usize, l: usize) -> bool {
        for i in 0..self.n_row_strategies() {
            if self.col_payoffs[(i, j)] <= self.col_payoffs[(i, l)] {
                return false;
            }
        }
        true
    }

    /// Iterated elimination of strictly dominated strategies (row player).
    /// Returns surviving row strategy indices.
    pub fn iesds_row(&self) -> HashSet<usize> {
        let n = self.n_row_strategies();
        let mut surviving: HashSet<usize> = (0..n).collect();
        let mut changed = true;
        while changed {
            changed = false;
            let surv: Vec<usize> = surviving.iter().copied().collect();
            for &k in &surv {
                let dominated = surv.iter().any(|&i| {
                    i != k && self.strictly_dominates_row(i, k)
                });
                if dominated {
                    surviving.remove(&k);
                    changed = true;
                    break;
                }
            }
        }
        surviving
    }

    /// Return the payoff pair at (i, j).
    pub fn payoff(&self, i: usize, j: usize) -> (f64, f64) {
        (self.row_payoffs[(i, j)], self.col_payoffs[(i, j)])
    }

    /// Compute expected payoff for the row player given mixed strategy profiles.
    /// `sigma_r` is the row player's mixed strategy (length = n_row_strategies).
    /// `sigma_c` is the column player's mixed strategy (length = n_col_strategies).
    pub fn expected_payoff_row(
        &self,
        sigma_r: &DVector<f64>,
        sigma_c: &DVector<f64>,
    ) -> f64 {
        let mut val = 0.0;
        for i in 0..self.n_row_strategies() {
            for j in 0..self.n_col_strategies() {
                val += sigma_r[i] * sigma_c[j] * self.row_payoffs[(i, j)];
            }
        }
        val
    }

    /// Compute expected payoff for the column player given mixed strategies.
    pub fn expected_payoff_col(
        &self,
        sigma_r: &DVector<f64>,
        sigma_c: &DVector<f64>,
    ) -> f64 {
        let mut val = 0.0;
        for i in 0..self.n_row_strategies() {
            for j in 0..self.n_col_strategies() {
                val += sigma_r[i] * sigma_c[j] * self.col_payoffs[(i, j)];
            }
        }
        val
    }
}

/// Convenience: Prisoner's Dilemma.
/// (C, D) — Cooperate, Defect.  T > R > P > S, 2R > T + S.
pub fn prisoners_dilemma(t: f64, r: f64, p: f64, s: f64) -> NormalFormGame {
    // Row = player 1 (C=0, D=1), Col = player 2 (C=0, D=1)
    NormalFormGame::new(
        DMatrix::from_row_slice(2, 2, &[r, s, t, p]),
        DMatrix::from_row_slice(2, 2, &[r, t, s, p]),
    )
}

/// Matching Pennies — a zero-sum game with no pure NE.
pub fn matching_pennies() -> NormalFormGame {
    NormalFormGame::new(
        DMatrix::from_row_slice(2, 2, &[1.0, -1.0, -1.0, 1.0]),
        DMatrix::from_row_slice(2, 2, &[-1.0, 1.0, 1.0, -1.0]),
    )
}

/// Battle of the Sexes.
pub fn battle_of_the_sexes() -> NormalFormGame {
    NormalFormGame::new(
        DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 0.0, 2.0]),
        DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]),
    )
}

/// Coordination game.
pub fn coordination_game() -> NormalFormGame {
    NormalFormGame::new(
        DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
        DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dvector;

    #[test]
    fn test_prisoners_dilemma_dominant() {
        let g = prisoners_dilemma(5.0, 3.0, 1.0, 0.0);
        // Defect (row 1) strictly dominates Cooperate (row 0)
        assert!(g.strictly_dominates_row(1, 0));
        // Cooperate does not dominate Defect
        assert!(!g.strictly_dominates_row(0, 1));
    }

    #[test]
    fn test_prisoners_dilemma_pure_ne() {
        let g = prisoners_dilemma(5.0, 3.0, 1.0, 0.0);
        let ne = g.pure_nash_equilibria();
        assert_eq!(ne, vec![(1, 1)]); // (Defect, Defect)
    }

    #[test]
    fn test_matching_pennies_no_pure_ne() {
        let g = matching_pennies();
        let ne = g.pure_nash_equilibria();
        assert!(ne.is_empty());
    }

    #[test]
    fn test_battle_of_the_sexes_two_ne() {
        let g = battle_of_the_sexes();
        let ne = g.pure_nash_equilibria();
        assert_eq!(ne.len(), 2);
        assert!(ne.contains(&(0, 0)));
        assert!(ne.contains(&(1, 1)));
    }

    #[test]
    fn test_coordination_game_two_ne() {
        let g = coordination_game();
        let ne = g.pure_nash_equilibria();
        assert_eq!(ne.len(), 2);
        assert!(ne.contains(&(0, 0)));
        assert!(ne.contains(&(1, 1)));
    }

    #[test]
    fn test_best_response_row() {
        let g = prisoners_dilemma(5.0, 3.0, 1.0, 0.0);
        // Against Cooperate (j=0): row payoffs are 3 (C) and 5 (D) → best = {1}
        assert_eq!(g.best_response_row(0), vec![1]);
        // Against Defect (j=1): row payoffs are 0 (C) and 1 (D) → best = {1}
        assert_eq!(g.best_response_row(1), vec![1]);
    }

    #[test]
    fn test_best_response_col() {
        let g = prisoners_dilemma(5.0, 3.0, 1.0, 0.0);
        // Against Cooperate (i=0): col payoffs are 3 (C) and 5 (D) → best = {1}
        assert_eq!(g.best_response_col(0), vec![1]);
        // Against Defect (i=1): col payoffs are 0 (C) and 1 (D) → best = {1}
        assert_eq!(g.best_response_col(1), vec![1]);
    }

    #[test]
    fn test_iesds() {
        let g = prisoners_dilemma(5.0, 3.0, 1.0, 0.0);
        let surv_row = g.iesds_row();
        assert!(surv_row.contains(&1));
        assert!(!surv_row.contains(&0));
    }

    #[test]
    fn test_expected_payoff() {
        let g = matching_pennies();
        let sr = dvector![0.5, 0.5];
        let sc = dvector![0.5, 0.5];
        let pr = g.expected_payoff_row(&sr, &sc);
        let pc = g.expected_payoff_col(&sr, &sc);
        assert!((pr - 0.0).abs() < 1e-10);
        assert!((pc - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_payoff_accessor() {
        let g = prisoners_dilemma(5.0, 3.0, 1.0, 0.0);
        let (p1, p2) = g.payoff(1, 1); // (Defect, Defect)
        assert!((p1 - 1.0).abs() < 1e-10);
        assert!((p2 - 1.0).abs() < 1e-10);
    }
}
