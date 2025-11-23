use itertools::izip;
use rand::prelude::*;
use std::array;

mod neuron;

use crate::sim::common::*;
use neuron::Neuron;

const WEIGHT_SUM: f64 = 78.0;

const WEIGHT_EI: f64 = 10.4;
const WEIGHT_IE: f64 = 17.0;

// ms
const TC_POST_LTD: f64 = 20.0;
const TC_POST_LTP: f64 = 40.0;
const NU_PRE: f64 = 0.0001;
const NU_POST: f64 = 0.01;

type Weights = [f64; IMAGE_SIZE];

#[derive(Debug, Clone)]
pub struct Cell {
    pub exc: Neuron,
    pub inh: Neuron,
    pub weights: Weights,
    traces: [(f64, f64); IMAGE_SIZE],
}

impl Cell {
    pub fn new(rng: &mut ThreadRng) -> Cell {
        let mut weights = array::from_fn(|_| rng.random_range(0.0..=1.0));
        Self::normalize(&mut weights);
        Cell {
            exc: Neuron::new(true),
            inh: Neuron::new(false),
            weights,
            traces: [(0.0, 0.0); IMAGE_SIZE],
        }
    }

    fn normalize(weights: &mut Weights) {
        let sum: f64 = weights.iter().sum();
        let r = WEIGHT_SUM / sum;
        weights.iter_mut().for_each(|w| *w *= r);
    }

    pub fn step(&mut self, dt: f64, test_mode: bool, inh_spikes: f64, input: &[f64; IMAGE_SIZE], x_pre: &[f64; IMAGE_SIZE]) -> (f64, f64) {
        // update exc neuron
        let dge = self.weights.iter().zip(input.iter()).map(|(&w, &i)| i * w).sum();
        let dgi = (inh_spikes - self.inh.s) * WEIGHT_IE;
        self.exc.step(test_mode, dt, dge, dgi);

        // update inh neuron
        let dge = self.exc.s * WEIGHT_EI;
        let dgi = 0.0;
        self.inh.step(test_mode, dt, dge, dgi);

        // update the weights using stdp
        if !test_mode {
            let decay_ltd = (-dt / TC_POST_LTD).exp();
            let decay_ltp = (-dt / TC_POST_LTP).exp();
            for (w, (ltd, ltp), &s_pre, &pre) in izip!(self.weights.iter_mut(), self.traces.iter_mut(), input.iter(), x_pre.iter()) {
                *ltd = self.exc.s + (1.0 - self.exc.s) * *ltd * decay_ltd;
                *ltp = self.exc.s + (1.0 - self.exc.s) * *ltp * decay_ltp;
                flush_to_zero(ltd);
                flush_to_zero(ltp);
                *w += -s_pre * NU_PRE * *ltd + self.exc.s * NU_POST * pre * *ltp;
                *w = w.min(1.0);
                // flush_to_zero(w);
            }
            Self::normalize(&mut self.weights);
        }
        (self.exc.s, self.inh.s)
    }
}
