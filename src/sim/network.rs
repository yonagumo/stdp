use itertools::izip;
use rand::prelude::*;

mod neuron;

use crate::sim::types::*;
use neuron::Neuron;

const WEIGHT_SUM: f64 = 78.0;
const TC_PRE: f64 = 20.0;
const TC_POST_LTD: f64 = 20.0;
const TC_POST_LTP: f64 = 40.0;
const NU_PRE: f64 = 0.0001;
const NU_POST: f64 = 0.01;

const WEIGHT_EI: f64 = 10.4;
const WEIGHT_IE: f64 = 17.0;

pub struct Network {
    rng: ThreadRng,
    input: Vec<bool>,
    weight: Vec<Vec<f64>>,
    exc: Vec<Neuron>,
    inh: Vec<Neuron>,
    x_pre: Vec<f64>,
    x_post: Vec<Vec<(f64, f64)>>,
}

impl Network {
    pub fn new(size: usize) -> Network {
        let mut rng = rand::rng();
        let mut weight = (0..size).map(|_| (0..IMAGE_SIZE).map(|_| rng.random_range(0.0..=1.0)).collect()).collect();
        Self::normalize(&mut weight);
        Network {
            rng,
            input: vec![false; IMAGE_SIZE],
            weight,
            exc: vec![Neuron::new(true); size],
            inh: vec![Neuron::new(false); size],
            x_pre: vec![0.0; IMAGE_SIZE],
            x_post: vec![vec![(0.0, 0.0); IMAGE_SIZE]; size],
        }
    }

    pub fn get_weight(&self) -> &Vec<Vec<f64>> {
        &self.weight
    }

    fn normalize(weight: &mut Vec<Vec<f64>>) {
        for weights in weight.iter_mut() {
            let sum: f64 = weights.iter().sum();
            let r = WEIGHT_SUM / sum;
            weights.iter_mut().for_each(|w| *w *= r);
        }
    }

    pub fn step(&mut self, test_mode: bool, dt: f64, image: &Image, intensity: f64) -> i32 {
        self.input.iter_mut().zip(image.iter()).for_each(|(b, v)| {
            let rate = *v as f64 / 4.0 * intensity; // Hz
            *b = self.rng.random_bool(rate * dt / 1000.0);
        });

        let mut spike_count = 0;

        // update exc neurons
        for (i, (neuron, weights)) in self.exc.iter_mut().zip(self.weight.iter()).enumerate() {
            let dge = weights.iter().zip(self.input.iter()).map(|(w, i)| if *i { *w } else { 0.0 }).sum();
            let dgi = self.inh.iter().enumerate().map(|(j, ni)| if ni.s && j != i { WEIGHT_IE } else { 0.0 }).sum();
            neuron.step(test_mode, dt, dge, dgi);
            if neuron.s {
                spike_count += 1;
            }
        }

        // update inh neurons
        for (neuron, exc) in self.inh.iter_mut().zip(&self.exc) {
            let dge = if exc.s { WEIGHT_EI } else { 0.0 };
            let dgi = 0.0;
            neuron.step(test_mode, dt, dge, dgi);
        }

        // update the weights using stdp
        if !test_mode {
            self.x_pre.iter_mut().zip(&self.input).for_each(|(x, s)| {
                *x = if *s { 1.0 } else { *x * (-dt / TC_PRE).exp() };
            });
            for (weights, traces, exc) in izip!(self.weight.iter_mut(), self.x_post.iter_mut(), self.exc.iter()) {
                for (w, s_pre, pre, (ltd, ltp)) in izip!(weights.iter_mut(), self.input.iter(), self.x_pre.iter(), traces.iter()) {
                    *w += if *s_pre { NU_PRE * ltd * -1.0 } else { 0.0 } + if exc.s { NU_POST * pre * ltp } else { 0.0 };
                    *w = w.min(1.0);
                }
                traces.iter_mut().for_each(|(ltd, ltp)| {
                    *ltd = if exc.s { 1.0 } else { *ltd * (-dt / TC_POST_LTD).exp() };
                    *ltp = if exc.s { 1.0 } else { *ltp * (-dt / TC_POST_LTP).exp() };
                });
            }

            Self::normalize(&mut self.weight);
        }

        return spike_count;
    }
}
