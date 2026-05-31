//! Evolutionary game theory: replicator dynamics, evolutionarily stable strategies (ESS).

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// An evolutionary game defined by a payoff matrix for a single population.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionaryGame {
    /// Payoff matrix A: A[(i,j)] = payoff to strategy i against strategy j.
    pub payoff_matrix: DMatrix<f64>,
}

impl EvolutionaryGame {
    /// Create a new evolutionary game.
    pub fn new(payoff_matrix: DMatrix<f64>) -> Self {
        Self { payoff_matrix }
    }

    /// Number of pure strategies.
    pub fn n_strategies(&self) -> usize {
        self.payoff_matrix.nrows()
    }

    /// Compute the expected payoff of strategy i against population x.
    pub fn payoff_against(&self, i: usize, x: &DVector<f64>) -> f64 {
        let mut val = 0.0;
        for j in 0..self.n_strategies() {
            val += self.payoff_matrix[(i, j)] * x[j];
        }
        val
    }

    /// Compute the average payoff of population x against itself.
    pub fn average_payoff(&self, x: &DVector<f64>) -> f64 {
        let mut val = 0.0;
        let n = self.n_strategies();
        for i in 0..n {
            for j in 0..n {
                val += x[i] * x[j] * self.payoff_matrix[(i, j)];
            }
        }
        val
    }

    /// One step of replicator dynamics: dx_i/dt = x_i * (f_i(x) - f̄(x))
    /// Returns the new population state.
    pub fn replicator_step(&self, x: &DVector<f64>, dt: f64) -> DVector<f64> {
        let n = self.n_strategies();
        let avg = self.average_payoff(x);

        let mut dx = DVector::zeros(n);
        for i in 0..n {
            let fi = self.payoff_against(i, x);
            dx[i] = x[i] * (fi - avg);
        }

        let new_x = x + dx.scale(dt);

        // Clamp to [0, 1] and renormalize
        let mut result = DVector::zeros(n);
        for i in 0..n {
            result[i] = new_x[i].max(0.0);
        }
        let sum: f64 = result.iter().sum();
        if sum > 0.0 {
            result.scale_mut(1.0 / sum);
        }
        result
    }

    /// Run replicator dynamics for multiple steps.
    pub fn replicator_dynamics(
        &self,
        x0: &DVector<f64>,
        dt: f64,
        steps: usize,
    ) -> Vec<DVector<f64>> {
        let mut trajectory = vec![x0.clone()];
        let mut x = x0.clone();
        for _ in 0..steps {
            x = self.replicator_step(&x, dt);
            trajectory.push(x.clone());
        }
        trajectory
    }

    /// Check if strategy x* is an Evolutionarily Stable Strategy (ESS).
    /// x* is ESS iff: for all x ≠ x*, either
    ///   (1) u(x*, x*) > u(x, x*), or
    ///   (2) u(x*, x*) = u(x, x*) and u(x*, x) > u(x, x)
    /// We check against pure strategy mutants.
    pub fn is_ess(&self, x_star: &DVector<f64>) -> bool {
        let n = self.n_strategies();
        let u_star_star = self.average_payoff(x_star);

        for i in 0..n {
            let mut mutant = DVector::zeros(n);
            mutant[i] = 1.0;

            // Skip if x* is pure strategy i
            let mut is_same = true;
            for j in 0..n {
                if (x_star[j] - mutant[j]).abs() > 1e-10 {
                    is_same = false;
                    break;
                }
            }
            if is_same {
                continue;
            }

            let u_mutant_star = self.payoff_against(i, x_star);

            if u_star_star > u_mutant_star + 1e-10 {
                // Condition 1 satisfied
                continue;
            }

            if (u_star_star - u_mutant_star).abs() < 1e-10 {
                // Check condition 2: u(x*, mutant) > u(mutant, mutant)
                let u_star_mutant: f64 = {
                    let mut val = 0.0;
                    for j in 0..n {
                        val += x_star[j] * self.payoff_matrix[(j, i)];
                    }
                    val
                };
                let u_mutant_mutant = self.payoff_matrix[(i, i)];

                if u_star_mutant > u_mutant_mutant + 1e-10 {
                    continue;
                }
            }

            return false;
        }
        true
    }

    /// Compute fitness landscape: payoff of each pure strategy against population x.
    pub fn fitness(&self, x: &DVector<f64>) -> DVector<f64> {
        let n = self.n_strategies();
        let mut f = DVector::zeros(n);
        for i in 0..n {
            f[i] = self.payoff_against(i, x);
        }
        f
    }
}

/// Hawk-Dove game: V = value of resource, C = cost of fighting.
pub fn hawk_dove_game(v: f64, c: f64) -> EvolutionaryGame {
    //         Hawk    Dove
    // Hawk   (V-C)/2   V
    // Dove    0        V/2
    EvolutionaryGame::new(DMatrix::from_row_slice(
        2,
        2,
        &[(v - c) / 2.0, v, 0.0, v / 2.0],
    ))
}

/// Coordination game for evolutionary setting.
pub fn evolutionary_coordination(a: f64, b: f64) -> EvolutionaryGame {
    //         A      B
    // A       a      0
    // B       0      b
    EvolutionaryGame::new(DMatrix::from_row_slice(2, 2, &[a, 0.0, 0.0, b]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dvector;

    #[test]
    fn test_hawk_dove_ess() {
        let game = hawk_dove_game(4.0, 6.0);
        // ESS: mixed strategy x* = V/C = 4/6 = 2/3 Hawk
        let x_star = dvector![2.0 / 3.0, 1.0 / 3.0];
        assert!(game.is_ess(&x_star));
    }

    #[test]
    fn test_hawk_dove_pure_not_ess() {
        let game = hawk_dove_game(4.0, 6.0);
        let pure_hawk = dvector![1.0, 0.0];
        let pure_dove = dvector![0.0, 1.0];
        assert!(!game.is_ess(&pure_hawk));
        assert!(!game.is_ess(&pure_dove));
    }

    #[test]
    fn test_coordination_both_ess() {
        let game = evolutionary_coordination(3.0, 2.0);
        let a = dvector![1.0, 0.0];
        let b = dvector![0.0, 1.0];
        assert!(game.is_ess(&a));
        assert!(game.is_ess(&b));
    }

    #[test]
    fn test_replicator_converges() {
        let game = evolutionary_coordination(3.0, 2.0);
        let x0 = dvector![0.6, 0.4];
        let trajectory = game.replicator_dynamics(&x0, 0.1, 100);
        let final_x = trajectory.last().unwrap();
        // Should converge toward strategy A (payoff 3 > 2)
        assert!(final_x[0] > 0.99);
    }

    #[test]
    fn test_fitness() {
        let game = hawk_dove_game(4.0, 6.0);
        let x = dvector![0.5, 0.5];
        let f = game.fitness(&x);
        // Hawk fitness: 0.5 * (-1) + 0.5 * 4 = 1.5
        // Dove fitness: 0.5 * 0 + 0.5 * 2 = 1.0
        assert!((f[0] - 1.5).abs() < 1e-10);
        assert!((f[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_average_payoff() {
        let game = hawk_dove_game(4.0, 6.0);
        let x = dvector![0.5, 0.5];
        let avg = game.average_payoff(&x);
        // = 0.25*(-1) + 0.25*4 + 0.25*0 + 0.25*2 = 1.25
        assert!((avg - 1.25).abs() < 1e-10);
    }

    #[test]
    fn test_replicator_step_normalizes() {
        let game = hawk_dove_game(4.0, 6.0);
        let x = dvector![0.5, 0.5];
        let new_x = game.replicator_step(&x, 0.5);
        let sum: f64 = new_x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(new_x[0] >= 0.0);
        assert!(new_x[1] >= 0.0);
    }
}
