pub mod mnist;

mod network;
mod types;

use network::Network;
use types::*;

// ms
const T: f64 = 350.0;
const T_REST: f64 = 150.0;
const DT: f64 = 1.0;

const NT: usize = (T / DT) as usize;
const NT_REST: usize = (T_REST / DT) as usize;

pub fn learn(images: Vec<Image>, size: [usize; 2]) {
    let mut network = Network::new(size);
    let blank = vec![0; IMAGE_SIZE];

    for (i, image) in images.iter().enumerate() {
        println!("learn: {:5} / {}", i + 1, images.len());
        for intensity in 1.. {
            let mut spike_count = 0;
            for _ in 0..NT {
                spike_count += network.step(DT, image, intensity as f64);
            }
            if spike_count >= 5 {
                break;
            }
        }

        for _ in 0..NT_REST {
            network.step(DT, &blank, 1.0);
        }
    }
}
