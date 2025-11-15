use rand::prelude::*;

mod neuron;

use crate::sim::types::*;
use neuron::Neuron;

const WEIGHT_SUM: f64 = 78.0;

pub struct Network {
    exc: Vec<Neuron>,
    inh: Vec<Neuron>,
    weight: Vec<Vec<f64>>,
    input: Vec<bool>,
    rng: ThreadRng,
}

impl Network {
    pub fn new(size: [usize; 2]) -> Network {
        let [w, h] = size;
        let exc = vec![Neuron::new(true); w * h];
        let inh = vec![Neuron::new(false); w * h];
        let mut rng = rand::rng();
        let mut weight = (0..w * h).map(|_| (0..IMAGE_SIZE).map(|_| rng.random_range(0.0..=1.0)).collect()).collect();
        Self::normalize(&mut weight);
        let input = vec![false; IMAGE_SIZE];
        Network {
            exc,
            inh,
            weight,
            input,
            rng,
        }
    }

    fn normalize(weight: &mut Vec<Vec<f64>>) {
        for weights in weight.iter_mut() {
            let sum: f64 = weights.iter().sum();
            let r = WEIGHT_SUM / sum;
            weights.iter_mut().for_each(|w| *w *= r);
        }
    }

    pub fn step(&mut self, dt: f64, image: &Image, intensity: f64) -> i32 {
        self.input.iter_mut().enumerate().for_each(|(i, b)| {
            let rate = image[i] as f64 / 4.0 * intensity; // Hz
            *b = self.rng.random_bool(rate * dt / 1000.0);
        });

        let mut spike_count = 0;

        // update exc neurons
        for (i, neuron) in self.exc.iter_mut().enumerate() {
            let dge = self.weight[i].iter().enumerate().map(|(i, w)| if self.input[i] { *w } else { 0.0 }).sum();
            let dgi = self.inh.iter().map(|ni| if ni.s { 1.0 } else { 0.0 }).sum();
            neuron.step(dt, dge, dgi);
            if neuron.s {
                spike_count += 1;
            }
        }

        // update inh neurons
        for (i, neuron) in self.inh.iter_mut().enumerate() {
            let dge = if self.exc[i].s { 1.0 } else { 0.0 };
            let dgi = 0.0;
            neuron.step(dt, dge, dgi);
        }

        // update the weights using stdp
        for (i, weights) in self.weight.iter_mut().enumerate() {}

        Self::normalize(&mut self.weight);

        return spike_count;
    }
}
