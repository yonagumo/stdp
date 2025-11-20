use itertools::izip;
use rand::prelude::*;
use std::array;

mod neuron;

use crate::sim::types::*;
use neuron::Neuron;

type Weights = [f64; IMAGE_SIZE];

const WEIGHT_SUM: f64 = 78.0;

const WEIGHT_EI: f64 = 10.4;
const WEIGHT_IE: f64 = 17.0;

// ms
const TC_POST_LTD: f64 = 20.0;
const TC_POST_LTP: f64 = 40.0;
const NU_PRE: f64 = 0.0001;
const NU_POST: f64 = 0.01;

#[derive(Debug, Clone)]
pub struct Cell {
    weights: Weights,
    exc: Neuron,
    inh: Neuron,
    traces: [(f64, f64); IMAGE_SIZE],
}

impl Cell {
    pub fn new(rng: &mut ThreadRng) -> Cell {
        let mut weights = array::from_fn(|_| rng.random_range(0.0..=1.0));
        Self::normalize(&mut weights);
        Cell {
            weights,
            exc: Neuron::new(true),
            inh: Neuron::new(false),
            traces: [(0.0, 0.0); IMAGE_SIZE],
        }
    }

    pub fn get_weights(&self) -> &Weights {
        &self.weights
    }

    fn normalize(weights: &mut Weights) {
        let sum: f64 = weights.iter().sum();
        let r = WEIGHT_SUM / sum;
        weights.iter_mut().for_each(|w| *w *= r);
    }

    pub fn step(
        &mut self,
        dt: f64,
        test_mode: bool,
        input: &[bool; IMAGE_SIZE],
        inh_spikes: usize,
        x_pre: &[f64; IMAGE_SIZE],
    ) -> (bool, bool) {
        // update exc neuron
        let dge = self.weights.iter().zip(input.iter()).map(|(w, i)| if *i { *w } else { 0.0 }).sum();
        let dgi = (inh_spikes - if self.inh.s { 1 } else { 0 }) as f64 * WEIGHT_IE;
        self.exc.step(test_mode, dt, dge, dgi);

        // update inh neuron
        let dge = if self.exc.s { WEIGHT_EI } else { 0.0 };
        let dgi = 0.0;
        self.inh.step(test_mode, dt, dge, dgi);

        // update the weights using stdp
        if !test_mode {
            for (w, (ltd, ltp), s_pre, pre) in izip!(self.weights.iter_mut(), self.traces.iter_mut(), input.iter(), x_pre.iter()) {
                *w += if *s_pre { NU_PRE * *ltd * -1.0 } else { 0.0 } + if self.exc.s { NU_POST * pre * *ltp } else { 0.0 };
                *w = w.min(1.0);
                *ltd = if self.exc.s { 1.0 } else { *ltd * (-dt / TC_POST_LTD).exp() };
                *ltp = if self.exc.s { 1.0 } else { *ltp * (-dt / TC_POST_LTP).exp() };
            }
            Self::normalize(&mut self.weights);
        }
        (self.exc.s, self.inh.s)
    }
}
