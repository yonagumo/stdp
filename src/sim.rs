use image::{ImageReader, Rgb, RgbImage};
use rayon::ThreadPoolBuilder;
use std::fs::File;
use std::io::{self, Read, Write};

pub mod mnist;

mod common;
mod network;

use common::*;
use network::Network;

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

pub fn label(path: &str, images: Vec<Image>, labels: Vec<Label>) {
    let pool = ThreadPoolBuilder::new().num_threads(N_THREADS).build().unwrap();
    let ([w, h], weights) = import(path);
    let size = w * h;
    let len_width = images.len().to_string().chars().count();

    let mut count_label = vec![0; 10];
    let mut count_fire = vec![[0.0; 10]; size];

    pool.install(|| {
        let network = Network::from_weights(weights);
        for (i, (image, label)) in images.iter().zip(&labels).enumerate() {
            let n = i + 1;
            println!("label: {:len_width$} / {}", n, images.len());
            count_label[*label as usize] += 1;
            let mut net = network.clone();
            for _ in 0..NT {
                net.step(true, DT, image, 1.0);
                for (s, c) in net.fired().iter().zip(count_fire.iter_mut()) {
                    if *s == TRUE {
                        c[*label as usize] += 1.0;
                    }
                }
            }
        }
    });

    for c in count_fire.iter_mut() {
        for (spikes, total) in c.iter_mut().zip(&count_label) {
            *spikes /= *total as f64;
        }
    }

    let result: Vec<usize> = count_fire
        .iter()
        .map(|rates| rates.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap().0)
        .collect();

    let mut total = [0; 10];
    result.iter().for_each(|l| total[*l] += 1);

    println!("input_labels: {count_label:?}");
    println!("cell_num: {total:?}");

    result.chunks(w).for_each(|row| println!("{row:?}"));

    let output: String = result.into_iter().map(|n| n.to_string()).collect();

    let base_path = &path[0..path.len() - 4];
    let output_path = format!("{base_path}.txt");
    let mut file = File::create(output_path).unwrap();
    write!(file, "{output}").unwrap();
}

fn import(path: &str) -> ([usize; 2], Vec<Weights>) {
    println!("import: {path}");
    let img = ImageReader::open(path).unwrap().decode().unwrap().into_rgb8();
    let w = img.width() as usize / IMAGE_WIDTH;
    let h = img.height() as usize / IMAGE_HEIGHT;
    let size = w * h;
    let mut weights = Vec::new();
    let mut cell_weights = [0.0; IMAGE_SIZE];
    for i in 0..size {
        let sx = i % w;
        let sy = i / w;
        for (j, ws) in cell_weights.iter_mut().enumerate() {
            let x = sx * IMAGE_WIDTH + j % IMAGE_WIDTH;
            let y = sy * IMAGE_HEIGHT + j / IMAGE_WIDTH;
            let rgb = img.get_pixel(x as u32, y as u32).0;
            let v = rgb[0];
            *ws = v as f64 / 255.0;
        }
        weights.push(cell_weights.clone());
    }
    ([w, h], weights)
}

pub fn test(path: &str, images: Vec<Image>, answers: Vec<Label>) {
    let pool = ThreadPoolBuilder::new().num_threads(N_THREADS).build().unwrap();
    let ([w, h], weights) = import(&format!("{path}.png"));
    let size = w * h;
    let len_width = images.len().to_string().chars().count();
    let labels = load_labels(&format!("{path}.txt"));

    let mut correct = 0;
    let mut matrix = [[0; 10]; 10];

    let mut count_label = vec![0; 10];

    for l in labels.iter() {
        count_label[*l as usize] += 1;
    }

    pool.install(|| {
        let network = Network::from_weights(weights);

        for (i, (image, &answer)) in images.iter().zip(&answers).enumerate() {
            let n = i + 1;
            print!("test: {:len_width$} / {}", n, images.len());
            let mut net = network.clone();
            let mut count_fire = vec![0; size];
            for _ in 0..NT {
                net.step(true, DT, image, 1.0);
                for (s, c) in net.fired().iter().zip(count_fire.iter_mut()) {
                    if *s == TRUE {
                        *c += 1;
                    }
                }
            }
            let mut spikes = [0; 10];
            for (s, l) in count_fire.iter().zip(&labels) {
                spikes[*l as usize] += s;
            }
            let rates: Vec<f64> = spikes.iter().zip(&count_label).map(|(&s, &n)| s as f64 / n as f64).collect();
            let result = rates.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap().0;
            matrix[answer as usize][result] += 1;
            let c = result == answer as usize;
            if c {
                correct += 1;
            }
            println!(" | {answer} -> {result} {}", if c { "o" } else { "x" });
        }
    });

    println!("done! correct: {correct} / {} ({}%)", images.len(), correct as f64 / images.len() as f64 * 100.0);

    for (a, exp) in matrix.iter().enumerate() {
        println!("{a}: {exp:?}");
    }
}

fn load_labels(path: &str) -> Vec<Label> {
    let mut labels = String::new();
    let mut f = File::open(path).unwrap();
    f.read_to_string(&mut labels).unwrap();
    labels.chars().map(|c| c.to_digit(10).unwrap() as Label).collect()
}
