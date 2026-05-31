//! Zero-sum games: minimax theorem, value of the game, saddle points.

use crate::normal_form::NormalFormGame;
use nalgebra::DMatrix;

/// A two-player zero-sum game where player 1's payoff is `payoffs` and
/// player 2's payoff is `-payoffs`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ZeroSumGame {
    pub payoffs: DMatrix<f64>,
}

impl ZeroSumGame {
    /// Create a zero-sum game from player 1's payoff matrix.
    pub fn new(payoffs: DMatrix<f64>) -> Self {
        Self { payoffs }
    }

    /// Convert from a NormalFormGame (must satisfy col_payoffs = -row_payoffs).
    pub fn from_normal_form(g: &NormalFormGame) -> Self {
        Self {
            payoffs: g.row_payoffs.clone(),
        }
    }

    /// Find saddle points (pure-strategy equilibria).
    /// A saddle point (i,j) satisfies: payoff[i,j] is the min of its row and max of its col.
    /// Actually: max of its column and min of its row for player 1 maximin.
    /// Returns (row_index, col_index, value).
    pub fn saddle_points(&self) -> Vec<(usize, usize, f64)> {
        let (m, n) = self.payoffs.shape();
        let mut row_mins: Vec<(usize, f64)> = vec![];
        for i in 0..m {
            let mut min_val = f64::INFINITY;
            for j in 0..n {
                if self.payoffs[(i, j)] < min_val {
                    min_val = self.payoffs[(i, j)];
                }
            }
            row_mins.push((i, min_val));
        }
        let mut col_maxes: Vec<(usize, f64)> = vec![];
        for j in 0..n {
            let mut max_val = f64::NEG_INFINITY;
            for i in 0..m {
                if self.payoffs[(i, j)] > max_val {
                    max_val = self.payoffs[(i, j)];
                }
            }
            col_maxes.push((j, max_val));
        }

        let lower = row_mins
            .iter()
            .map(|&(_, v)| v)
            .fold(f64::NEG_INFINITY, f64::max);
        let upper = col_maxes
            .iter()
            .map(|&(_, v)| v)
            .fold(f64::INFINITY, f64::min);

        let mut points = vec![];
        if (lower - upper).abs() < 1e-10 {
            for i in 0..m {
                for j in 0..n {
                    if (self.payoffs[(i, j)] - lower).abs() < 1e-10 {
                        points.push((i, j, lower));
                    }
                }
            }
        }
        points
    }

    /// Compute the maximin value (lower value) of the game for player 1.
    pub fn maximin_value(&self) -> f64 {
        let (m, n) = self.payoffs.shape();
        let mut best = f64::NEG_INFINITY;
        for i in 0..m {
            let mut row_min = f64::INFINITY;
            for j in 0..n {
                if self.payoffs[(i, j)] < row_min {
                    row_min = self.payoffs[(i, j)];
                }
            }
            if row_min > best {
                best = row_min;
            }
        }
        best
    }

    /// Compute the minimax value (upper value) of the game for player 1.
    pub fn minimax_value(&self) -> f64 {
        let (m, n) = self.payoffs.shape();
        let mut best = f64::INFINITY;
        for j in 0..n {
            let mut col_max = f64::NEG_INFINITY;
            for i in 0..m {
                if self.payoffs[(i, j)] > col_max {
                    col_max = self.payoffs[(i, j)];
                }
            }
            if col_max < best {
                best = col_max;
            }
        }
        best
    }

    /// Value of the game if saddle points exist (equal to maximin = minimax).
    /// Returns None if no pure saddle point.
    pub fn value_pure(&self) -> Option<f64> {
        let lower = self.maximin_value();
        let upper = self.minimax_value();
        if (lower - upper).abs() < 1e-10 {
            Some(lower)
        } else {
            None
        }
    }

    /// Solve a 2x2 zero-sum game analytically using mixed strategies.
    /// Returns ((p, 1-p), (q, 1-q), value) or None if pure saddle point exists.
    #[allow(clippy::type_complexity)]
    pub fn solve_2x2(&self) -> Option<((f64, f64), (f64, f64), f64)> {
        let (m, n) = self.payoffs.shape();
        assert!(m == 2 && n == 2, "Must be a 2x2 game");

        if self.value_pure().is_some() {
            return None; // Pure saddle point exists
        }

        let a = self.payoffs[(0, 0)];
        let b = self.payoffs[(0, 1)];
        let c = self.payoffs[(1, 0)];
        let d = self.payoffs[(1, 1)];

        let denom = (a + d) - (b + c);
        if denom.abs() < 1e-12 {
            return None;
        }

        let p = (d - c) / denom; // Prob player 1 plays row 0
        let q = (d - b) / denom; // Prob player 2 plays col 0
        let val = (a * d - b * c) / denom;

        Some(((p, 1.0 - p), (q, 1.0 - q), val))
    }

    /// Number of row strategies.
    pub fn n_row(&self) -> usize {
        self.payoffs.nrows()
    }

    /// Number of column strategies.
    pub fn n_col(&self) -> usize {
        self.payoffs.ncols()
    }

    /// Dominant strategy for player 1 (if exists). Returns the row index.
    pub fn dominant_row(&self) -> Option<usize> {
        let (m, n) = self.payoffs.shape();
        'outer: for i in 0..m {
            for k in 0..m {
                if i == k {
                    continue;
                }
                for j in 0..n {
                    if self.payoffs[(i, j)] <= self.payoffs[(k, j)] {
                        continue 'outer;
                    }
                }
            }
            return Some(i);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;

    #[test]
    fn test_saddle_point() {
        // Simple game with saddle point at (0, 1)
        let g = ZeroSumGame::new(dmatrix![1.0, 3.0; -2.0, 10.0]);
        let sp = g.saddle_points();
        assert!(!sp.is_empty());
        // Row mins: row0=1, row1=-2 → maximin = 1
        // Col maxes: col0=1, col1=10 → minimax = 1
        // Saddle at (0,0) where value=1 and (0,1) should NOT be saddle
        // Actually row0 min = 1 (at col0), col0 max = 1 (at row0) → saddle at (0,0)
        assert!(sp.iter().any(|&(i, j, _)| i == 0 && j == 0));
    }

    #[test]
    fn test_matching_pennies_no_saddle() {
        let g = ZeroSumGame::new(dmatrix![1.0, -1.0; -1.0, 1.0]);
        let sp = g.saddle_points();
        assert!(sp.is_empty());
        assert!(g.value_pure().is_none());
    }

    #[test]
    fn test_matching_pennies_2x2() {
        let g = ZeroSumGame::new(dmatrix![1.0, -1.0; -1.0, 1.0]);
        let sol = g.solve_2x2();
        assert!(sol.is_some());
        let ((p, _p2), (q, _q2), val) = sol.unwrap();
        assert!((p - 0.5).abs() < 1e-10);
        assert!((q - 0.5).abs() < 1e-10);
        assert!(val.abs() < 1e-10);
    }

    #[test]
    fn test_maximin_minimax() {
        let g = ZeroSumGame::new(dmatrix![2.0, -1.0; -1.0, 1.0]);
        // Row mins:  -1, -1 → maximin = -1
        // Col maxes: 2, 1   → minimax = 1
        assert!((g.maximin_value() - (-1.0)).abs() < 1e-10);
        assert!((g.minimax_value() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_sum_value_pure() {
        let g = ZeroSumGame::new(dmatrix![3.0, 1.0; 2.0, -1.0]);
        // Row mins: 1, -1 → maximin = 1
        // Col maxes: 3, 1 → minimax = 1
        let v = g.value_pure();
        assert!(v.is_some());
        assert!((v.unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dominant_row() {
        // Row 0 dominates row 1
        let g = ZeroSumGame::new(dmatrix![3.0, 2.0; 1.0, 0.0]);
        assert_eq!(g.dominant_row(), Some(0));
    }
}
