use std::fs;

use crate::sim::common::*;

pub fn load_mnist_images(path: &str) -> Vec<Image> {
    let data = fs::read(path).unwrap();
    let magic_number = u32::from_be_bytes(take4(&data, 0));
    let num_images = u32::from_be_bytes(take4(&data, 4));
    let height = u32::from_be_bytes(take4(&data, 8));
    let width = u32::from_be_bytes(take4(&data, 12));
    println!("load: {} images ({}x{}, magic number: 0x{:08x})", num_images, width, height, magic_number);
    data[16..].chunks((width * height) as usize).map(|bs| bs.to_vec()).collect()
}

pub fn load_mnist_labels(path: &str) -> Vec<Label> {
    let data = fs::read(path).unwrap();
    let magic_number = u32::from_be_bytes(take4(&data, 0));
    let num_labels = u32::from_be_bytes(take4(&data, 4));
    println!("load: {} labels (magic number: {:08x})", num_labels, magic_number);
    data[8..].to_vec()
}

fn take4<T: Copy>(a: &[T], offset: usize) -> [T; 4] {
    [a[offset], a[offset + 1], a[offset + 2], a[offset + 3]]
}
