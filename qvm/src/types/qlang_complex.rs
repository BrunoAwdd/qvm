use num_complex::Complex64;
use num_traits::{One, Zero};
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, Div, MulAssign, DivAssign, Neg};
use ndarray::{Array2, ArrayBase, Data, Ix2, ScalarOperand};

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct QLangComplex {
    pub re: f64,
    pub im: f64,
}

impl QLangComplex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn norm_sqr(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn arg(&self) -> f64 {
        self.im.atan2(self.re)
    }

    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    pub fn zero() -> Self {
        QLangComplex { re: 0.0, im: 0.0 }
    }

    pub fn one() -> Self {
        QLangComplex::new(1.0, 0.0)
    }
    
    pub fn neg_one() -> Self {
        QLangComplex::new(-1.0, 0.0)
    }

    pub fn i() -> Self {
        QLangComplex::new(0.0, 1.0)
    }

    pub fn neg_i() -> Self {
        QLangComplex::new(0.0, -1.0)
    }
}

impl From<Complex64> for QLangComplex {
    fn from(c: Complex64) -> Self {
        Self { re: c.re, im: c.im }
    }
}

impl From<QLangComplex> for Complex64 {
    fn from(c: QLangComplex) -> Self {
        Complex64::new(c.re, c.im)
    }
}

pub fn to_complex64<A: Data<Elem = QLangComplex>>(a: &ArrayBase<A, Ix2>) -> Array2<Complex64> {
    a.map(|z| Complex64::from(*z))
}

pub fn from_complex64<A: Data<Elem = Complex64>>(a: &ArrayBase<A, Ix2>) -> Array2<QLangComplex> {
    a.map(|z| QLangComplex::from(*z))
}

pub fn to_complex_vec(src: &[QLangComplex]) -> Vec<Complex64> {
    src.iter().copied().map(Complex64::from).collect()
}

impl Neg for QLangComplex {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl Add for QLangComplex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { re: self.re + rhs.re, im: self.im + rhs.im }
    }
}

impl AddAssign for QLangComplex {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Sub for QLangComplex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { re: self.re - rhs.re, im: self.im - rhs.im }
    }
}

impl SubAssign for QLangComplex {
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl Mul<f64> for QLangComplex {
    type Output = QLangComplex;

    fn mul(self, rhs: f64) -> QLangComplex {
        QLangComplex {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

impl Mul<QLangComplex> for f64 {
    type Output = QLangComplex;

    fn mul(self, rhs: QLangComplex) -> QLangComplex {
        QLangComplex {
            re: rhs.re * self,
            im: rhs.im * self,
        }
    }
}

impl Mul for QLangComplex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl MulAssign for QLangComplex {
    fn mul_assign(&mut self, rhs: Self) {
        let temp = *self * rhs;
        *self = temp;
    }
}

impl Div for QLangComplex {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / denom,
            im: (self.im * rhs.re - self.re * rhs.im) / denom,
        }
    }
}

impl Div<f64> for QLangComplex {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self {
            re: self.re / rhs,
            im: self.im / rhs,
        }
    }
}

impl Div<f64> for &QLangComplex {
    type Output = QLangComplex;

    fn div(self, rhs: f64) -> Self::Output {
        QLangComplex::new(self.re / rhs, self.im / rhs)
    }
}

impl DivAssign for QLangComplex {
    fn div_assign(&mut self, rhs: Self) {
        let temp = *self / rhs;
        *self = temp;
    }
}

impl Mul<QLangComplex> for Complex64 {
    type Output = Complex64;
    fn mul(self, rhs: QLangComplex) -> Complex64 {
        self * Complex64::new(rhs.re, rhs.im)
    }
}

impl PartialEq for QLangComplex {
    fn eq(&self, other: &Self) -> bool {
        (self.re - other.re).abs() < 1e-10 && (self.im - other.im).abs() < 1e-10
    }
}

impl Zero for QLangComplex {
    fn zero() -> Self {
        QLangComplex { re: 0.0, im: 0.0 }
    }

    fn is_zero(&self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }
}

impl One for QLangComplex {
    fn one() -> Self {
        QLangComplex::new(1.0, 0.0)
    }

    fn is_one(&self) -> bool {
        self.re == 1.0 && self.im == 0.0
    }
}

impl ScalarOperand for QLangComplex {}

#[cfg(feature = "cuda")]
unsafe impl cust::memory::DeviceCopy for QLangComplex {}