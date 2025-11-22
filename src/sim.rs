use image::{Rgb, RgbImage};
use rayon::ThreadPoolBuilder;
use std::fs::File;
use std::io::{self, Write};

pub mod mnist;

mod network;
mod types;

use network::Network;
use types::*;

const N_THREADS: usize = 8;

// ms
const T: f64 = 350.0;
const T_REST: f64 = 150.0;
const DT: f64 = 0.5;

const NT: usize = (T / DT) as usize;
const NT_REST: usize = (T_REST / DT) as usize;

const MIN_SPIKES: usize = 5;
const INTENSITY_MAX: usize = 8;

pub fn learn(output_path: &str, images: Vec<Image>, [w, h]: [usize; 2]) {
    let pool = ThreadPoolBuilder::new().num_threads(N_THREADS).build().unwrap();
    let size = w * h;
    let len_width = images.len().to_string().chars().count();
    let blank = vec![0; IMAGE_SIZE];
    let mut cycle = 0;
    let mut miss = 0;

    let weights = pool.install(|| {
        let mut network = Network::new(size);
        export(&format!("{output_path}0.png"), &network.get_weights(), [w, h]);
        for (i, image) in images.iter().enumerate() {
            let n = i + 1;
            print!("\nlearn: {:len_width$} / {}", n, images.len());
            io::stdout().flush().unwrap();
            print!(" | intensity, spikes:");
            for intensity in 1.. {
                cycle += 1;
                let mut spike_count = 0;
                for _ in 0..NT {
                    let (exc, _inh) = network.step(false, DT, image, intensity as f64);
                    spike_count += exc;
                }
                print!(" ({intensity},{spike_count:2})");
                if spike_count >= MIN_SPIKES {
                    break;
                } else if intensity == INTENSITY_MAX {
                    miss += 1;
                    break;
                }
                //io::stdout().flush().unwrap();
            }
            for _ in 0..NT_REST {
                network.step(false, DT, &blank, 1.0);
            }
            if n % 100 == 0 && n != images.len() {
                println!("");
                export(&format!("{output_path}{n}.png"), &network.get_weights(), [w, h]);
            }
        }
        network.get_weights()
    });

    println!("");

    println!("cycle: {cycle}, miss: {miss}");
    export(&format!("{output_path}result.png"), &weights, [w, h]);
    println!("");
}

fn export(path: &str, weight: &Vec<Vec<f64>>, [w, h]: [usize; 2]) {
    print!("export: {}", path);
    // io::stdout().flush().unwrap();
    let mut img = RgbImage::new((IMAGE_WIDTH * w) as u32, (IMAGE_HEIGHT * h) as u32);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let nx = x as usize / IMAGE_WIDTH;
        let ny = y as usize / IMAGE_HEIGHT;
        let sx = x as usize % IMAGE_WIDTH;
        let sy = y as usize % IMAGE_HEIGHT;
        let v = (weight[nx + w * ny][sx + IMAGE_WIDTH * sy] * 256.0).floor().min(255.0) as u8;
        *pixel = Rgb([v, v, v]);
    }
    img.save(path).unwrap();
    let mut file = File::create("latest.txt").unwrap();
    write!(file, "{path}").unwrap();
}
