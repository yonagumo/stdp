// leaky integrate-and-fire model

use std::fmt;

use crate::sim::common::*;

// ms
const TAU_GE: f64 = 1.0;
const TAU_GI: f64 = 2.0;
const TAU_THETA: f64 = 1e7;

struct LIFParams {
    // mV
    v_rest: f64,
    v_reset: f64,
    v_thresh: f64,
    e_exc: f64,
    e_inh: f64,
    // ms
    tau: f64,
    refrac: f64,
}

const PARAMS_EXC: LIFParams = LIFParams {
    v_rest: -65.0,
    v_reset: -65.0,
    v_thresh: -52.0,
    e_exc: 0.0,
    e_inh: -100.0,
    tau: 100.0,
    refrac: 5.0,
};

const PARAMS_INH: LIFParams = LIFParams {
    v_rest: -60.0,
    v_reset: -45.0,
    v_thresh: -40.0,
    e_exc: 0.0,
    e_inh: -85.0,
    tau: 10.0,
    refrac: 2.0,
};

#[derive(Clone, Debug)]
pub struct Neuron {
    pub s: f64, // 0.0 or 1.0
    exc: bool,
    v: f64,
    theta: f64,
    g_e: f64,
    g_i: f64,
    refr: f64,
}

impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:+8.4}", if self.s == TRUE { 0.0 } else { self.v })
    }
}

impl Neuron {
    pub fn new(excitatory: bool) -> Neuron {
        Neuron {
            s: FALSE,
            exc: excitatory,
            v: if excitatory { PARAMS_EXC.v_reset } else { PARAMS_INH.v_reset } - 40.0,
            theta: 20.0,
            g_e: 0.0,
            g_i: 0.0,
            refr: 0.0,
        }
    }

    pub fn step(&mut self, test_mode: bool, dt: f64, dge: f64, dgi: f64) {
        let p = if self.exc { PARAMS_EXC } else { PARAMS_INH };
        self.g_e = self.g_e * (-dt / TAU_GE).exp() + dge;
        self.g_i = self.g_i * (-dt / TAU_GI).exp() + dgi;
        flush_to_zero(&mut self.g_e);
        flush_to_zero(&mut self.g_i);
        // self.v += ((p.v_rest - self.v) + (p.e_exc - self.v) * self.g_e + (p.e_inh - self.v) * self.g_i) / p.tau * dt;
        let g_tot = 1.0 + self.g_e + self.g_i;
        let v_inf = (p.v_rest + p.e_exc * self.g_e + p.e_inh * self.g_i) / g_tot;
        let tau_eff = p.tau / g_tot;
        self.v = v_inf + (self.v - v_inf) * (-dt / tau_eff).exp();
        self.s = if self.refr <= 0.0 && self.v > self.theta - 20.0 + p.v_thresh { TRUE } else { FALSE };
        if self.exc && !test_mode {
            self.theta = self.theta * (-dt / TAU_THETA).exp() + self.s * 0.05;
        }
        if self.s == TRUE {
            self.v = p.v_reset;
            self.refr = p.refrac;
        }
        self.refr = (self.refr - dt).max(0.0);
    }
}
