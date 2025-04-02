use num_complex::Complex64;
use num_traits::Zero;
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, Div, MulAssign, DivAssign};
use ndarray::ScalarOperand;
use num_traits::One;

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct CudaComplex {
    pub re: f64,
    pub im: f64,
}

impl CudaComplex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn norm_sqr(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn arg(&self) -> f64 {
        self.im.atan2(self.re)
    }
}

impl From<Complex64> for CudaComplex {
    fn from(c: Complex64) -> Self {
        Self { re: c.re, im: c.im }
    }
}

impl From<CudaComplex> for Complex64 {
    fn from(c: CudaComplex) -> Self {
        Complex64::new(c.re, c.im)
    }
}

pub fn to_complex_vec(src: &[CudaComplex]) -> Vec<Complex64> {
    src.iter().copied().map(Complex64::from).collect()
}


#[cfg(feature = "cuda")]
unsafe impl cust::memory::DeviceCopy for CudaComplex {}


impl Zero for CudaComplex {
    fn zero() -> Self {
        CudaComplex { re: 0.0, im: 0.0 }
    }

    fn is_zero(&self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }
}

impl Add for CudaComplex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { re: self.re + rhs.re, im: self.im + rhs.im }
    }
}

impl AddAssign for CudaComplex {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Sub for CudaComplex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { re: self.re - rhs.re, im: self.im - rhs.im }
    }
}

impl SubAssign for CudaComplex {
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl Mul for CudaComplex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl MulAssign for CudaComplex {
    fn mul_assign(&mut self, rhs: Self) {
        let temp = *self * rhs;
        *self = temp;
    }
}

impl Div for CudaComplex {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / denom,
            im: (self.im * rhs.re - self.re * rhs.im) / denom,
        }
    }
}

impl DivAssign for CudaComplex {
    fn div_assign(&mut self, rhs: Self) {
        let temp = *self / rhs;
        *self = temp;
    }
}

impl Mul<CudaComplex> for Complex64 {
    type Output = Complex64;
    fn mul(self, rhs: CudaComplex) -> Complex64 {
        self * Complex64::new(rhs.re, rhs.im)
    }
}

impl PartialEq for CudaComplex {
    fn eq(&self, other: &Self) -> bool {
        (self.re - other.re).abs() < 1e-10 && (self.im - other.im).abs() < 1e-10
    }
}

impl One for CudaComplex {
    fn one() -> Self {
        CudaComplex::new(1.0, 0.0)
    }

    fn is_one(&self) -> bool {
        self.re == 1.0 && self.im == 0.0
    }
}

impl ScalarOperand for CudaComplex {}
