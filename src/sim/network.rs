use itertools::izip;
use rand::prelude::*;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::iter;

mod cell;

use crate::sim::common::*;
use cell::Cell;

// ms
const TC_PRE: f64 = 20.0;

#[derive(Debug)]
pub struct Network {
    rng: ThreadRng,
    cells: Vec<Cell>,
    input: [f64; IMAGE_SIZE],
    x_pre: [f64; IMAGE_SIZE],
    inh_spikes: f64,
}

impl Network {
    pub fn new(size: usize) -> Network {
        let mut rng = rand::rng();
        let cells = iter::repeat_with(|| Cell::new(&mut rng)).take(size).collect();

        Network {
            rng,
            cells,
            input: [FALSE; IMAGE_SIZE],
            x_pre: [0.0; IMAGE_SIZE],
            inh_spikes: 0.0,
        }
    }

    // pub fn debug(&self) -> String {
    //     self.cells[0].exc.to_string()
    // }

    pub fn get_weights(&self) -> Vec<Vec<f64>> {
        self.cells.iter().map(|c| c.weights.to_vec()).collect()
    }

    pub fn step(&mut self, test_mode: bool, dt: f64, image: &Image, intensity: f64) -> (usize, usize) {
        let decay_x_pre = (-dt / TC_PRE).exp();
        for (v, s, x) in izip!(image, self.input.iter_mut(), self.x_pre.iter_mut()) {
            let rate = *v as f64 / 4.0 * intensity; // Hz
            *s = if self.rng.random_bool(rate * dt / 1000.0) { TRUE } else { FALSE };
            if !test_mode {
                *x = *x * decay_x_pre + *s;
                flush_to_zero(x);
            }
        }

        // update cells
        let (exc_spikes, inh_spikes) = self
            .cells
            .par_iter_mut()
            .fold(
                || (0.0, 0.0),
                |(exc, inh), cell| {
                    let (e, i) = cell.step(dt, test_mode, self.inh_spikes, &self.input, &self.x_pre);
                    (exc + e, inh + i)
                },
            )
            .reduce(|| (0.0, 0.0), |(exc, inh), (e, i)| (exc + e, inh + i));
        self.inh_spikes = inh_spikes;

        return (exc_spikes as usize, inh_spikes as usize);
    }
}
