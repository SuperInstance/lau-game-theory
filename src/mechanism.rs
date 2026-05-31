//! Mechanism design: Vickrey auction, incentive compatibility.

use serde::{Deserialize, Serialize};

/// Outcome of an auction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuctionOutcome {
    /// Winner index (None if no winner).
    pub winner: Option<usize>,
    /// Price paid by the winner.
    pub price: f64,
    /// Revenue collected.
    pub revenue: f64,
}

/// Run a second-price sealed-bid (Vickrey) auction.
/// The highest bidder wins, pays the second-highest bid.
pub fn vickrey_auction(bids: &[f64]) -> AuctionOutcome {
    if bids.is_empty() {
        return AuctionOutcome {
            winner: None,
            price: 0.0,
            revenue: 0.0,
        };
    }

    let mut best_idx = 0;
    let mut best_bid = bids[0];
    let mut second_bid = f64::NEG_INFINITY;

    for (i, &bid) in bids.iter().enumerate().skip(1) {
        if bid > best_bid {
            second_bid = best_bid;
            best_bid = bid;
            best_idx = i;
        } else if bid > second_bid {
            second_bid = bid;
        }
    }

    let price = if second_bid == f64::NEG_INFINITY {
        0.0 // Only one bidder
    } else {
        second_bid
    };

    AuctionOutcome {
        winner: Some(best_idx),
        price,
        revenue: price,
    }
}

/// Run a first-price sealed-bid auction.
pub fn first_price_auction(bids: &[f64]) -> AuctionOutcome {
    if bids.is_empty() {
        return AuctionOutcome {
            winner: None,
            price: 0.0,
            revenue: 0.0,
        };
    }

    let mut best_idx = 0;
    let mut best_bid = bids[0];

    for (i, &bid) in bids.iter().enumerate() {
        if bid > best_bid {
            best_bid = bid;
            best_idx = i;
        }
    }

    AuctionOutcome {
        winner: Some(best_idx),
        price: best_bid,
        revenue: best_bid,
    }
}

/// A direct revelation mechanism for allocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectMechanism {
    /// Number of agents.
    pub n_agents: usize,
    /// Number of items.
    pub n_items: usize,
}

impl DirectMechanism {
    /// Create a new direct mechanism.
    pub fn new(n_agents: usize, n_items: usize) -> Self {
        Self { n_agents, n_items }
    }

    /// VCG (Vickrey-Clarke-Groves) mechanism for a single item.
    /// Each agent reports a value. The mechanism allocates efficiently
    /// and charges the externality.
    pub fn vcg_single_item(&self, reported_values: &[f64]) -> Vec<(bool, f64)> {
        assert_eq!(reported_values.len(), self.n_agents);

        let mut winner = 0;
        let mut best = reported_values[0];
        let mut second = f64::NEG_INFINITY;

        for (i, &v) in reported_values.iter().enumerate() {
            if v > best {
                second = best;
                best = v;
                winner = i;
            } else if v > second {
                second = v;
            }
        }

        let second_best = if second == f64::NEG_INFINITY {
            0.0
        } else {
            second
        };

        let mut result = vec![(false, 0.0); self.n_agents];
        result[winner] = (true, second_best);
        result
    }
}

/// Check if a mechanism is incentive compatible (truthful bidding is dominant).
/// Tests by comparing truthful utility vs. misreport utility for all agents.
pub fn check_incentive_compatibility<F>(n_agents: usize, n_values: usize, mechanism: F) -> bool
where
    F: Fn(&[f64]) -> Vec<(bool, f64)>,
{
    // Test with random-ish value profiles
    let test_values: Vec<Vec<f64>> = (0..n_values)
        .map(|i| {
            (0..n_agents)
                .map(|j| ((i * 7 + j * 13 + 1) % 100) as f64 / 10.0)
                .collect()
        })
        .collect();

    for values in &test_values {
        let truthful_outcome = mechanism(values);

        for agent in 0..n_agents {
            let truthful_utility = if truthful_outcome[agent].0 {
                values[agent] - truthful_outcome[agent].1
            } else {
                0.0
            };

            // Try misreports
            for misreport in &[0.0, 0.5, 1.0, 5.0, 10.0] {
                let mut mis_values = values.clone();
                mis_values[agent] = *misreport;
                let mis_outcome = mechanism(&mis_values);

                let mis_utility = if mis_outcome[agent].0 {
                    values[agent] - mis_outcome[agent].1
                } else {
                    0.0
                };

                if mis_utility > truthful_utility + 1e-10 {
                    return false;
                }
            }
        }
    }
    true
}

/// Run an all-pay auction: everyone pays their bid, highest bidder wins.
pub fn all_pay_auction(bids: &[f64]) -> AuctionOutcome {
    if bids.is_empty() {
        return AuctionOutcome {
            winner: None,
            price: 0.0,
            revenue: 0.0,
        };
    }

    let mut best_idx = 0;
    let mut best_bid = bids[0];
    let mut total_paid = 0.0;

    for (i, &bid) in bids.iter().enumerate() {
        total_paid += bid;
        if bid > best_bid {
            best_bid = bid;
            best_idx = i;
        }
    }

    AuctionOutcome {
        winner: Some(best_idx),
        price: best_bid,
        revenue: total_paid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vickrey_basic() {
        let outcome = vickrey_auction(&[10.0, 20.0, 15.0]);
        assert_eq!(outcome.winner, Some(1));
        assert!((outcome.price - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_vickrey_two_bidders() {
        let outcome = vickrey_auction(&[5.0, 3.0]);
        assert_eq!(outcome.winner, Some(0));
        assert!((outcome.price - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_vickrey_single_bidder() {
        let outcome = vickrey_auction(&[42.0]);
        assert_eq!(outcome.winner, Some(0));
        assert!((outcome.price - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_vickrey_empty() {
        let outcome = vickrey_auction(&[]);
        assert_eq!(outcome.winner, None);
    }

    #[test]
    fn test_vickrey_ic() {
        assert!(check_incentive_compatibility(3, 10, |bids| {
            let outcome = vickrey_auction(bids);
            bidders_to_allocation(bids.len(), &outcome)
        }));
    }

    #[test]
    fn test_first_price_not_ic() {
        // First-price is NOT incentive compatible (you want to shade your bid)
        // With enough test profiles, this should fail
        let result = check_incentive_compatibility(2, 20, |bids| {
            let outcome = first_price_auction(bids);
            bidders_to_allocation(bids.len(), &outcome)
        });
        // First-price auction is NOT IC — so this should be false
        assert!(!result);
    }

    #[test]
    fn test_vcg_single_item() {
        let mech = DirectMechanism::new(3, 1);
        let result = mech.vcg_single_item(&[10.0, 30.0, 20.0]);
        assert!(result[1].0); // Player 1 wins
        assert!((result[1].1 - 20.0).abs() < 1e-10); // Pays second price
    }

    #[test]
    fn test_vcg_ic() {
        let mech = DirectMechanism::new(3, 1);
        assert!(check_incentive_compatibility(3, 10, |bids| {
            mech.vcg_single_item(bids)
        }));
    }

    #[test]
    fn test_all_pay() {
        let outcome = all_pay_auction(&[5.0, 8.0, 3.0]);
        assert_eq!(outcome.winner, Some(1));
        assert!((outcome.revenue - 16.0).abs() < 1e-10);
    }

    fn bidders_to_allocation(n: usize, outcome: &AuctionOutcome) -> Vec<(bool, f64)> {
        let mut result = vec![(false, 0.0); n];
        if let Some(w) = outcome.winner {
            result[w] = (true, outcome.price);
        }
        result
    }
}
