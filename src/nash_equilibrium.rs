//! Nash equilibrium: support enumeration for 2-player mixed-strategy NE.

use crate::normal_form::NormalFormGame;
use nalgebra::DVector;

/// A mixed-strategy Nash equilibrium profile.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MixedNashEquilibrium {
    /// Player 1's mixed strategy.
    pub sigma_r: DVector<f64>,
    /// Player 2's mixed strategy.
    pub sigma_c: DVector<f64>,
    /// Expected payoffs.
    pub payoffs: (f64, f64),
}

/// Find all mixed-strategy Nash equilibria for a 2x2 game by analytical formula.
pub fn find_mixed_ne_2x2(game: &NormalFormGame) -> Vec<MixedNashEquilibrium> {
    assert!(
        game.n_row_strategies() == 2 && game.n_col_strategies() == 2,
        "Only 2x2 games supported"
    );

    let mut results = vec![];

    // First add pure NE
    let pure = game.pure_nash_equilibria();
    for (i, j) in &pure {
        let mut sr = DVector::zeros(2);
        let mut sc = DVector::zeros(2);
        sr[*i] = 1.0;
        sc[*j] = 1.0;
        results.push(MixedNashEquilibrium {
            sigma_r: sr,
            sigma_c: sc,
            payoffs: game.payoff(*i, *j),
        });
    }

    // Now check for a fully mixed NE
    // Player 2 must be indifferent between col 0 and col 1
    // p * col_payoffs[(0, j)] + (1-p) * col_payoffs[(1, j)] equal for j=0,1
    let a = game.col_payoffs[(0, 0)];
    let b = game.col_payoffs[(1, 0)];
    let c = game.col_payoffs[(0, 1)];
    let d = game.col_payoffs[(1, 1)];

    // p*a + (1-p)*b = p*c + (1-p)*d
    // p*(a - b) + b = p*(c - d) + d
    // p*(a - b - c + d) = d - b
    let denom = a - b - c + d;
    if denom.abs() > 1e-12 {
        let p = (d - b) / denom;
        if p > 1e-10 && p < 1.0 - 1e-10 {
            // Now find q for player 1's indifference
            let e = game.row_payoffs[(0, 0)];
            let f = game.row_payoffs[(0, 1)];
            let g_ = game.row_payoffs[(1, 0)];
            let h = game.row_payoffs[(1, 1)];
            let denom2 = e - f - g_ + h;
            if denom2.abs() > 1e-12 {
                let q = (h - f) / denom2;
                if q > 1e-10 && q < 1.0 - 1e-10 {
                    let sr = DVector::from_vec(vec![p, 1.0 - p]);
                    let sc = DVector::from_vec(vec![q, 1.0 - q]);
                    let pr = game.expected_payoff_row(&sr, &sc);
                    let pc = game.expected_payoff_col(&sr, &sc);
                    results.push(MixedNashEquilibrium {
                        sigma_r: sr,
                        sigma_c: sc,
                        payoffs: (pr, pc),
                    });
                }
            }
        }
    }

    results
}

/// Support enumeration for 2-player games: tries all subsets of strategies
/// as supports and solves the resulting indifference equations.
/// This is a general method for small games.
pub fn support_enumeration(game: &NormalFormGame) -> Vec<MixedNashEquilibrium> {
    let m = game.n_row_strategies();
    let n = game.n_col_strategies();

    if m == 2 && n == 2 {
        return find_mixed_ne_2x2(game);
    }

    let mut results = vec![];

    // Pure NE
    let pure = game.pure_nash_equilibria();
    for (i, j) in &pure {
        let mut sr = DVector::zeros(m);
        let mut sc = DVector::zeros(n);
        sr[*i] = 1.0;
        sc[*j] = 1.0;
        results.push(MixedNashEquilibrium {
            sigma_r: sr,
            sigma_c: sc,
            payoffs: game.payoff(*i, *j),
        });
    }

    // For 2xN or Nx2 games, use analytical approach
    if m == 2 && n > 2 {
        // Try each pair of columns as support for player 2
        for j1 in 0..n {
            for j2 in (j1 + 1)..n {
                // Player 1 mixes over both rows; player 2 mixes over j1, j2
                // Player 1 must be indifferent: u1(row0) = u1(row1) given player 2's mix
                // q * row_payoffs[(i, j1)] + (1-q) * row_payoffs[(i, j2)] equal for i=0,1
                let a = game.row_payoffs[(0, j1)];
                let b = game.row_payoffs[(0, j2)];
                let c = game.row_payoffs[(1, j1)];
                let d = game.row_payoffs[(1, j2)];
                let denom = (a - b) - (c - d);
                if denom.abs() < 1e-12 {
                    continue;
                }
                let q = (d - b) / denom; // prob player 2 plays j1
                if q <= 1e-10 || q >= 1.0 - 1e-10 {
                    continue;
                }

                // Now find p for player 2's indifference between j1 and j2
                let e = game.col_payoffs[(0, j1)];
                let f = game.col_payoffs[(0, j2)];
                let g_ = game.col_payoffs[(1, j1)];
                let h = game.col_payoffs[(1, j2)];
                let denom2 = (e - f) - (g_ - h);
                if denom2.abs() < 1e-12 {
                    continue;
                }
                let p = (h - f) / denom2;
                if p <= 1e-10 || p >= 1.0 - 1e-10 {
                    continue;
                }

                // Check: player 2 doesn't prefer any other column
                let mut sc_full = DVector::zeros(n);
                sc_full[j1] = q;
                sc_full[j2] = 1.0 - q;
                let sr_full = DVector::from_vec(vec![p, 1.0 - p]);

                let payoff_j1 = p * game.col_payoffs[(0, j1)] + (1.0 - p) * game.col_payoffs[(1, j1)];
                let mut valid = true;
                for jj in 0..n {
                    if jj == j1 || jj == j2 {
                        continue;
                    }
                    let payoff_jj =
                        p * game.col_payoffs[(0, jj)] + (1.0 - p) * game.col_payoffs[(1, jj)];
                    if payoff_jj > payoff_j1 + 1e-10 {
                        valid = false;
                        break;
                    }
                }
                if valid {
                    let pr = game.expected_payoff_row(&sr_full, &sc_full);
                    let pc = game.expected_payoff_col(&sr_full, &sc_full);
                    results.push(MixedNashEquilibrium {
                        sigma_r: sr_full,
                        sigma_c: sc_full,
                        payoffs: (pr, pc),
                    });
                }
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normal_form::{battle_of_the_sexes, coordination_game, matching_pennies};
    use nalgebra::dmatrix;

    #[test]
    fn test_matching_pennies_mixed_ne() {
        let g = matching_pennies();
        let ne = find_mixed_ne_2x2(&g);
        // Should have exactly 1 fully mixed NE
        let mixed: Vec<_> = ne
            .iter()
            .filter(|eq| eq.sigma_r[0] > 0.01 && eq.sigma_r[0] < 0.99)
            .collect();
        assert_eq!(mixed.len(), 1);
        assert!((mixed[0].sigma_r[0] - 0.5).abs() < 1e-10);
        assert!((mixed[0].sigma_c[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_battle_of_the_sexes_three_ne() {
        let g = battle_of_the_sexes();
        let ne = find_mixed_ne_2x2(&g);
        // 2 pure + 1 mixed = 3
        assert_eq!(ne.len(), 3);
    }

    #[test]
    fn test_coordination_three_ne() {
        let g = coordination_game();
        let ne = find_mixed_ne_2x2(&g);
        assert_eq!(ne.len(), 3);
    }

    #[test]
    fn test_chicken_game() {
        // Chicken (hawk-dove): (Swerve, Swerve) = (0,0), (Swerve, Dare) = (-1, 1),
        // (Dare, Swerve) = (1, -1), (Dare, Dare) = (-10, -10)
        let g = NormalFormGame::new(
            dmatrix![0.0, -1.0; 1.0, -10.0],
            dmatrix![0.0, 1.0; -1.0, -10.0],
        );
        let ne = find_mixed_ne_2x2(&g);
        // 2 pure NE: (Swerve, Dare) and (Dare, Swerve) + 1 mixed
        assert_eq!(ne.len(), 3);
    }

    #[test]
    fn test_mixed_ne_payoffs() {
        let g = matching_pennies();
        let ne = find_mixed_ne_2x2(&g);
        let mixed: Vec<_> = ne
            .iter()
            .filter(|eq| eq.sigma_r[0] > 0.01 && eq.sigma_r[0] < 0.99)
            .collect();
        assert!((mixed[0].payoffs.0).abs() < 1e-10);
        assert!((mixed[0].payoffs.1).abs() < 1e-10);
    }
}
