pub type Image = Vec<u8>;
pub type Label = u8;

pub const IMAGE_WIDTH: usize = 28;
pub const IMAGE_HEIGHT: usize = 28;
pub const IMAGE_SIZE: usize = IMAGE_WIDTH * IMAGE_HEIGHT;

pub const FALSE: f64 = 0.0;
pub const TRUE: f64 = 1.0;

const MIN_NORMAL: f64 = f64::MIN_POSITIVE * 100.0;

pub fn flush_to_zero(v: &mut f64) {
    if *v < MIN_NORMAL {
        *v = 0.0;
    }
}
