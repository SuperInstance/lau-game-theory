//! # lau-game-theory
//!
//! A comprehensive Rust game theory library covering normal-form games,
//! Nash equilibrium, zero-sum games, extensive-form games, Bayesian games,
//! cooperative games, mechanism design, evolutionary game theory, and
//! multi-agent applications.

#![deny(unsafe_code)]

pub mod bayesian;
pub mod cooperative;
pub mod evolutionary;
pub mod extensive_form;
pub mod mechanism;
pub mod multi_agent;
pub mod nash_equilibrium;
pub mod normal_form;
pub mod zero_sum;

pub use bayesian::BayesianGame;
pub use cooperative::CooperativeGame;
pub use evolutionary::EvolutionaryGame;
pub use extensive_form::{backward_induction, BackwardInductionResult, GameNode, Player};
pub use mechanism::{vickrey_auction, AuctionOutcome, DirectMechanism};
pub use multi_agent::MultiAgentGame;
pub use nash_equilibrium::MixedNashEquilibrium;
pub use normal_form::NormalFormGame;
pub use zero_sum::ZeroSumGame;
