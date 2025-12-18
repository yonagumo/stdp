#![allow(dead_code, unused_variables)]
use chrono::Local;
use std::env;
use std::fs;
use std::io::Read;

mod sim;

use sim::mnist::{load_mnist_images, load_mnist_labels};

const MNIST_TRAIN_IMAGE_PATH: &str = "mnist/train-images.idx3-ubyte";
const MNIST_TRAIN_LABEL_PATH: &str = "mnist/train-labels.idx1-ubyte";
const MNIST_TEST_IMAGE_PATH: &str = "mnist/t10k-images.idx3-ubyte";
const MNIST_TEST_LABEL_PATH: &str = "mnist/t10k-labels.idx1-ubyte";

fn main() {
    let args: Vec<String> = env::args().collect();
    if let Some(task) = args.get(1) {
        match task.as_str() {
            "full" => todo!(),
            "learn" => learn(&args[2..]),
            "label" => label(&args[2..]),
            "test" => test(&args[2..]),
            _ => panic!("incorrect task"),
        }
    } else {
        panic!("specify task");
    }
}

fn learn(args: &[String]) {
    let images = load_mnist_images(MNIST_TRAIN_IMAGE_PATH);
    let (w, h) = args[0].split_once('x').unwrap();
    let (w, h) = (w.parse().unwrap(), h.parse().unwrap());
    let limit = args[1].parse().unwrap();
    let images = images.into_iter().cycle().take(limit).collect();
    let path = Local::now().format("output/%Y-%m-%d_%H-%M-%S/").to_string();
    fs::create_dir_all(&path).unwrap();
    sim::learn(&path, images, [w, h]);
}

fn label(args: &[String]) {
    let limit = args[1].parse().unwrap();
    let images = load_mnist_images(MNIST_TRAIN_IMAGE_PATH).into_iter().cycle().take(limit).collect();
    let labels = load_mnist_labels(MNIST_TRAIN_LABEL_PATH).into_iter().cycle().take(limit).collect();
    let mut latest = String::new();
    if let Ok(mut f) = fs::File::open("latest.txt") {
        f.read_to_string(&mut latest).unwrap();
    };
    let path = if args[0] == "latest" { latest } else { args[0].clone() };
    sim::label(&path, images, labels);
}

fn test(args: &[String]) {
    let limit = args[1].parse().unwrap();
    let images = load_mnist_images(MNIST_TEST_IMAGE_PATH).into_iter().cycle().take(limit).collect();
    let labels = load_mnist_labels(MNIST_TEST_LABEL_PATH).into_iter().cycle().take(limit).collect();
    let mut latest = String::new();
    if let Ok(mut f) = fs::File::open("latest.txt") {
        f.read_to_string(&mut latest).unwrap();
    };
    let path = if args[0] == "latest" { latest } else { args[0].clone() };
    sim::test(&path, images, labels);
}
