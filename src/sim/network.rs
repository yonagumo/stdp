use itertools::izip;
use rand::prelude::*;
use std::iter;

mod cell;

use crate::sim::types::*;
use cell::Cell;

// ms
const TC_PRE: f64 = 20.0;

pub struct Network {
    rng: ThreadRng,
    input: [bool; IMAGE_SIZE],
    x_pre: [f64; IMAGE_SIZE],
    cell: Vec<Cell>,
    inh_spikes: usize,
}

impl Network {
    pub fn new(size: usize) -> Network {
        let mut rng = rand::rng();
        let cell = iter::repeat_with(|| Cell::new(&mut rng)).take(size).collect();
        Network {
            rng,
            input: [false; IMAGE_SIZE],
            x_pre: [0.0; IMAGE_SIZE],
            cell,
            inh_spikes: 0,
        }
    }

    pub fn get_weights(&self) -> Vec<Vec<f64>> {
        self.cell.iter().map(|c| c.get_weights().to_vec()).collect()
    }

    pub fn step(&mut self, test_mode: bool, dt: f64, image: &Image, intensity: f64) -> (usize, usize) {
        for (v, s, x) in izip!(image, self.input.iter_mut(), self.x_pre.iter_mut()) {
            let rate = *v as f64 / 4.0 * intensity; // Hz
            *s = self.rng.random_bool(rate * dt / 1000.0);
            if !test_mode {
                *x = if *s { 1.0 } else { *x * (-dt / TC_PRE).exp() };
            }
        }

        let mut exc_spikes = 0;
        let mut inh_spikes_new = 0;
        for cell in self.cell.iter_mut() {
            let (ecx, inh) = cell.step(dt, test_mode, &self.input, self.inh_spikes, &self.x_pre);
            exc_spikes += if ecx { 1 } else { 0 };
            inh_spikes_new += if inh { 1 } else { 0 };
        }
        self.inh_spikes = inh_spikes_new;

        return (exc_spikes, inh_spikes_new);
    }
}
