pub mod iter;
mod compat;

pub use compat::*;

use std::ops::{Add, Sub, Div, Mul, Rem};

use thiserror::Error;

pub type Unit = usize;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum VarArray {
    Zero([Unit; 0]),
    One([Unit; 1]),
    Two([Unit; 2]),
    Three([Unit; 3]),
    Four([Unit; 4]),
    Five([Unit; 5]),
    Six([Unit; 6]),
    Seven([Unit; 7]),
    Eight([Unit; 8]),
    Etc(Box<[Unit]>),
}

impl VarArray {
    pub fn len(&self) -> usize {
        match self {
            VarArray::Zero(_) => 0,
            VarArray::One(_) => 1,
            VarArray::Two(_) => 2,
            VarArray::Three(_) => 3,
            VarArray::Four(_) => 4,
            VarArray::Five(_) => 5,
            VarArray::Six(_) => 6,
            VarArray::Seven(_) => 7,
            VarArray::Eight(_) => 8,
            VarArray::Etc(arr) => arr.len(),
        }
    }

    pub fn get(&self, index: usize) -> Result<Unit, VarArrayError> {
        match self {
            VarArray::Zero(_) => Err(VarArrayError::IndexOutOfRange { index, length: 0 }),
            VarArray::One(arr) => match index {
                0 => Ok(arr[0]),
                _ => Err(VarArrayError::IndexOutOfRange { index, length: 1 }),
            },
            VarArray::Two(arr) => match index {
                0 => Ok(arr[0]),
                1 => Ok(arr[1]),
                _ => Err(VarArrayError::IndexOutOfRange { index, length: 2 }),

            },
            VarArray::Three(arr) => match index {
                0 => Ok(arr[0]),
                1 => Ok(arr[1]),
                2 => Ok(arr[2]),
                _ => Err(VarArrayError::IndexOutOfRange { index, length: 3 }),
            },
            VarArray::Four(arr) => match index {
                0 => Ok(arr[0]),
                1 => Ok(arr[1]),
                2 => Ok(arr[2]),
                3 => Ok(arr[3]),
                _ => Err(VarArrayError::IndexOutOfRange { index, length: 4 }),
            },
            VarArray::Five(arr) => match index {
                0 => Ok(arr[0]),
                1 => Ok(arr[1]),
                2 => Ok(arr[2]),
                3 => Ok(arr[3]),
                4 => Ok(arr[4]),
                _ => Err(VarArrayError::IndexOutOfRange { index, length: 5 }),
            },
            VarArray::Six(arr) => match index {
                0 => Ok(arr[0]),
                1 => Ok(arr[1]),
                2 => Ok(arr[2]),
                3 => Ok(arr[3]),
                4 => Ok(arr[4]),
                5 => Ok(arr[5]),
                _ => Err(VarArrayError::IndexOutOfRange { index, length: 6 }),
            },
            VarArray::Seven(arr) => match index {
                0 => Ok(arr[0]),
                1 => Ok(arr[1]),
                2 => Ok(arr[2]),
                3 => Ok(arr[3]),
                4 => Ok(arr[4]),
                5 => Ok(arr[5]),
                6 => Ok(arr[6]),
                _ => Err(VarArrayError::IndexOutOfRange { index, length: 7 }),
            },
            VarArray::Eight(arr) =>  match index {
                0 => Ok(arr[0]),
                1 => Ok(arr[1]),
                2 => Ok(arr[2]),
                3 => Ok(arr[3]),
                4 => Ok(arr[4]),
                5 => Ok(arr[5]),
                6 => Ok(arr[6]),
                7 => Ok(arr[7]),
                _ => Err(VarArrayError::IndexOutOfRange { index, length: 8 }),
            },
            VarArray::Etc(arr) => arr.get(index).copied().ok_or(VarArrayError::IndexOutOfRange { index, length: arr.len() }),
        }
    }

    pub fn get_mut(&mut self, index: usize) -> Result<&mut Unit, VarArrayError> {
        match self {
            VarArray::Zero(_) => Err(VarArrayError::IndexOutOfRange { index, length: 0 }),
            VarArray::One(arr) => match index {
                0 => Ok(&mut arr[0]),
                _ => Err(VarArrayError::IndexOutOfRange { index, length: 1 }),
            },
            VarArray::Two(arr) => match index {
                0 => Ok(&mut arr[0]),
                1 => Ok(&mut arr[1]),
                _ => Err(VarArrayError::IndexOutOfRange { index, length: 2 }),

            },
            VarArray::Three(arr) => match index {
                0 => Ok(&mut arr[0]),
                1 => Ok(&mut arr[1]),
                2 => Ok(&mut arr[2]),
                _ => Err(VarArrayError::IndexOutOfRange { index, length: 3 }),

            },
            VarArray::Four(arr) => match index {
                0 => Ok(&mut arr[0]),
                1 => Ok(&mut arr[1]),
                2 => Ok(&mut arr[2]),
                3 => Ok(&mut arr[3]),
                _ => Err(VarArrayError::IndexOutOfRange { index, length: 4 }),

            },
            VarArray::Five(arr) => match index {
                0 => Ok(&mut arr[0]),
                1 => Ok(&mut arr[1]),
                2 => Ok(&mut arr[2]),
                3 => Ok(&mut arr[3]),
                4 => Ok(&mut arr[4]),
                _ => Err(VarArrayError::IndexOutOfRange { index, length: 5 }),

            },
            VarArray::Six(arr) => match index {
                0 => Ok(&mut arr[0]),
                1 => Ok(&mut arr[1]),
                2 => Ok(&mut arr[2]),
                3 => Ok(&mut arr[3]),
                4 => Ok(&mut arr[4]),
                5 => Ok(&mut arr[5]),
                _ => Err(VarArrayError::IndexOutOfRange { index, length: 6 }),

            },
            VarArray::Seven(arr) => match index {
                0 => Ok(&mut arr[0]),
                1 => Ok(&mut arr[1]),
                2 => Ok(&mut arr[2]),
                3 => Ok(&mut arr[3]),
                4 => Ok(&mut arr[4]),
                5 => Ok(&mut arr[5]),
                6 => Ok(&mut arr[6]),
                _ => Err(VarArrayError::IndexOutOfRange { index, length: 7 }),

            },
            VarArray::Eight(arr) =>  match index {
                0 => Ok(&mut arr[0]),
                1 => Ok(&mut arr[1]),
                2 => Ok(&mut arr[2]),
                3 => Ok(&mut arr[3]),
                4 => Ok(&mut arr[4]),
                5 => Ok(&mut arr[5]),
                6 => Ok(&mut arr[6]),
                7 => Ok(&mut arr[7]),
                _ => Err(VarArrayError::IndexOutOfRange { index, length: 8 }),

            },
            VarArray::Etc(arr) => {
                let length = arr.len();
                arr.get_mut(index).ok_or(VarArrayError::IndexOutOfRange { index, length })
            }
        }
    }

    pub fn iter(&self) -> Iter {
        Iter::new(self)
    }

    pub fn as_slice(&self) -> &[usize] {
        match self {
            VarArray::Zero(arr) => arr.as_slice(),
            VarArray::One(arr) => arr.as_slice(),
            VarArray::Two(arr) => arr.as_slice(),
            VarArray::Three(arr) => arr.as_slice(),
            VarArray::Four(arr) => arr.as_slice(),
            VarArray::Five(arr) => arr.as_slice(),
            VarArray::Six(arr) => arr.as_slice(),
            VarArray::Seven(arr) => arr.as_slice(),
            VarArray::Eight(arr) => arr.as_slice(),
            VarArray::Etc(arr) => arr,
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [usize] {
        match self {
            VarArray::Zero(arr) => arr.as_mut_slice(),
            VarArray::One(arr) => arr.as_mut_slice(),
            VarArray::Two(arr) => arr.as_mut_slice(),
            VarArray::Three(arr) => arr.as_mut_slice(),
            VarArray::Four(arr) => arr.as_mut_slice(),
            VarArray::Five(arr) => arr.as_mut_slice(),
            VarArray::Six(arr) => arr.as_mut_slice(),
            VarArray::Seven(arr) => arr.as_mut_slice(),
            VarArray::Eight(arr) => arr.as_mut_slice(),
            VarArray::Etc(arr) => arr,
        }
    }

    pub fn pointwise_two<F: Fn(Unit, Unit) -> Unit>(&self, rhs: &VarArray, op: F) -> Result<VarArray, VarArrayError> {
        match self {
            VarArray::Zero(_) => match rhs {
                VarArray::Zero(_) => Ok(VarArray::Zero([])),
                _ => Err(VarArrayError::LengthMismatch(self.len(), rhs.len()))
            },
            VarArray::One(a) => match rhs {
                VarArray::One(b) => Ok(VarArray::One([op(a[0], b[0])])),
                _ => Err(VarArrayError::LengthMismatch(self.len(), rhs.len()))
            },
            VarArray::Two(a) => match rhs {
                VarArray::Two(b) => Ok(VarArray::Two([op(a[0], b[0]), op(a[1], b[1])])),
                _ => Err(VarArrayError::LengthMismatch(self.len(), rhs.len()))
            },
            VarArray::Three(a) => match rhs {
                VarArray::Three(b) => Ok(VarArray::Three([op(a[0], b[0]), op(a[1], b[1]), op(a[2], b[2])])),
                _ => Err(VarArrayError::LengthMismatch(self.len(), rhs.len()))
            },
            VarArray::Four(a) => match rhs {
                VarArray::Four(b) => Ok(VarArray::Four([op(a[0], b[0]), op(a[1], b[1]), op(a[2], b[2]), op(a[3], b[3])])),
                _ => Err(VarArrayError::LengthMismatch(self.len(), rhs.len()))
            },
            VarArray::Five(a) => match rhs {
                VarArray::Five(b) => Ok(VarArray::Five([op(a[0], b[0]), op(a[1], b[1]), op(a[2], b[2]), op(a[3], b[3]), op(a[4], b[4])])),
                _ => Err(VarArrayError::LengthMismatch(self.len(), rhs.len()))
            },
            VarArray::Six(a) => match rhs {
                VarArray::Six(b) => Ok(VarArray::Six([op(a[0], b[0]), op(a[1], b[1]), op(a[2], b[2]), op(a[3], b[3]), op(a[4], b[4]), op(a[5], b[5])])),
                _ => Err(VarArrayError::LengthMismatch(self.len(), rhs.len()))
            },
            VarArray::Seven(a) => match rhs {
                VarArray::Seven(b) => Ok(VarArray::Seven([op(a[0], b[0]), op(a[1], b[1]), op(a[2], b[2]), op(a[3], b[3]), op(a[4], b[4]), op(a[5], b[5]), op(a[6], b[6])])),
                _ => Err(VarArrayError::LengthMismatch(self.len(), rhs.len()))
            },
            VarArray::Eight(a) => match rhs {
                VarArray::Eight(b) => Ok(VarArray::Eight([op(a[0], b[0]), op(a[1], b[1]), op(a[2], b[2]), op(a[3], b[3]), op(a[4], b[4]), op(a[5], b[5]), op(a[6], b[6]), op(a[7], b[7])])),
                _ => Err(VarArrayError::LengthMismatch(self.len(), rhs.len()))
            },
            VarArray::Etc(a) => match rhs {
                VarArray::Etc(b) => if a.len() == b.len() {
                    Ok(VarArray::Etc(a.iter().copied().zip(b.iter().copied()).map(|(a_e, b_e)| op(a_e, b_e)).collect()))
                } else {
                    Err(VarArrayError::LengthMismatch(self.len(), rhs.len()))
                },
                _ => Err(VarArrayError::LengthMismatch(self.len(), rhs.len()))
            }
        }
    }

    pub fn add(&self, rhs: &VarArray) -> Result<VarArray, VarArrayError> {
        self.pointwise_two(rhs, Unit::add)
    }

    pub fn sub(&self, rhs: &VarArray) -> Result<VarArray, VarArrayError> {
        self.pointwise_two(rhs, Unit::sub)
    }

    pub fn div(&self, rhs: &VarArray) -> Result<VarArray, VarArrayError> {
        self.pointwise_two(rhs, Unit::div)
    }

    pub fn mul(&self, rhs: &VarArray) -> Result<VarArray, VarArrayError> {
        self.pointwise_two(rhs, Unit::mul)
    }

    pub fn rem(&self, rhs: &VarArray) -> Result<VarArray, VarArrayError> {
        self.pointwise_two(rhs, Unit::rem)
    }
}

impl From<&[Unit]> for VarArray {
    fn from(value: &[Unit]) -> Self {
        match value.len() {
            0 => Self::Zero([]),
            1 => Self::One([value[0]]),
            2 => Self::Two([value[0], value[1]]),
            3 => Self::Three([value[0], value[1], value[2]]),
            4 => Self::Four([value[0], value[1], value[2], value[3]]),
            5 => Self::Five([value[0], value[1], value[2], value[3], value[4]]),
            6 => Self::Six([value[0], value[1], value[2], value[3], value[4], value[5]]),
            7 => Self::Seven([value[0], value[1], value[2], value[3], value[4], value[5], value[6]]),
            8 => Self::Eight([value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7]]),
            _ => Self::Etc(value.into()),
        }
    }
}

impl FromIterator<Unit> for VarArray {
    fn from_iter<T: IntoIterator<Item = Unit>>(iter: T) -> Self {
        let mut iter = iter.into_iter().collect::<Vec<Unit>>().into_iter();

        match iter.len() {
            0 => Self::Zero([]),
            1 => Self::One([iter.next().unwrap()]),
            2 => Self::Two([iter.next().unwrap(), iter.next().unwrap()]),
            3 => Self::Three([iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap()]),
            4 => Self::Four([iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap()]),
            5 => Self::Five([iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap()]),
            6 => Self::Six([iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap()]),
            7 => Self::Seven([iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap()]),
            8 => Self::Eight([iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap()]),
            _ => Self::Etc(iter.collect()),
        }
    }
}

impl VarArrayCompatible for VarArray {
    fn new(varr: VarArray) -> Self {
        varr
    }

    fn vararray(&self) -> &VarArray {
        self
    }

    fn vararray_mut(&mut self) -> &mut VarArray {
        self
    }
}

#[derive(Error, Debug, PartialEq)]
pub enum VarArrayError {
    #[error("Index out of range: {index} not in {length}")]
    IndexOutOfRange {
        index: usize,
        length: usize,
    },
    #[error("Length mismatch: {0} and {1}")]
    LengthMismatch(usize, usize),
}

macro_rules! varr {
    () => {
        VarArray::from([].as_slice())
    };
    ($($x:expr),+) => {
        VarArray::from([$($x),+].as_slice())
    };
}
pub(crate) use varr;

use self::iter::Iter;

#[cfg(test)]
mod test {
    use super::*;

    fn gen_varrs_with(limit: usize, inst_sample: &[Unit]) -> Vec<VarArray> {
        let mut out = Vec::<VarArray>::new();

        for curr_limit in 0..limit {
            out.push(VarArray::from(&inst_sample[0..=curr_limit]));
        }

        out
    }

    #[test]
    pub fn create_and_length() {
        let v0 = varr![];
        assert_eq!(v0.len(), 0);

        let v1 = varr![0];
        assert_eq!(v1.len(), 1);

        let v2 = varr![0, 0];
        assert_eq!(v2.len(), 2);

        let v3 = varr![0, 0, 0];
        assert_eq!(v3.len(), 3);

        let v4 = varr![0, 0, 0, 0];
        assert_eq!(v4.len(), 4);

        let v5 = varr![0, 0, 0, 0, 0];
        assert_eq!(v5.len(), 5);

        let v6 = varr![0, 0, 0, 0, 0, 0];
        assert_eq!(v6.len(), 6);

        let v7 = varr![0, 0, 0, 0, 0, 0, 0];
        assert_eq!(v7.len(), 7);

        let v8 = varr![0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(v8.len(), 8);

        let v9 = varr![0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(v9.len(), 9);

        let v10 = VarArray::from([0; 128].as_slice());
        assert_eq!(v10.len(), 128);
    }

    #[test]
    pub fn get() {
        let limit = 32usize;
        let sample = (0..limit).collect::<Box<[Unit]>>();

        let varrs = gen_varrs_with(limit, &sample);

        for varr in varrs {
            let curr_len = varr.len();

            for i in 0..limit {
                if i < curr_len {
                    assert_eq!(varr.get(i).unwrap(), *sample.get(i).unwrap());
                } else {
                    assert_eq!(varr.get(i), Result::Err(VarArrayError::IndexOutOfRange { index: i, length: varr.len() }));
                }
            }
        }
    }

    #[test]
    pub fn get_mut() {
        let limit = 32usize;
        let sample = (0..limit).collect::<Box<[Unit]>>();

        let mut varrs = gen_varrs_with(limit, &sample);

        for varr in varrs.iter_mut() {
            let curr_len = varr.len();

            for i in 0..limit {
                if i < curr_len {
                    *varr.get_mut(i).unwrap() = varr.get(i).unwrap() * 2;
                }
            }
        }

        for varr in varrs {
            let curr_len = varr.len();

            for i in 0..limit {
                if i < curr_len {
                    assert_eq!(varr.get(i).unwrap(), *sample.get(i).unwrap() * 2);
                } else {
                    assert_eq!(varr.get(i), Result::Err(VarArrayError::IndexOutOfRange { index: i, length: varr.len() }));
                }
            }
        }
    }

    #[test]
    pub fn iter() {
        let limit = 32usize;
        let sample = (0..limit).collect::<Box<[Unit]>>();

        let varrs = gen_varrs_with(limit, &sample);

        for varr in varrs {
            let curr_len = varr.len();

            let zipped = varr.iter().zip(sample.iter().copied()).collect::<Box<[(Unit, Unit)]>>();

            assert_eq!(zipped.len(), curr_len);
            assert!(zipped.iter().all(|(t, o)| t == o));
        }
    }

    #[test]
    pub fn add() {
        let limit = 32usize;
        let sample_forwards = (0..limit).collect::<Box<[Unit]>>();
        let sample_backwards = (0..limit*2).step_by(2).rev().collect::<Box<[Unit]>>();

        let varrs_forwards = gen_varrs_with(limit, &sample_forwards);
        let varrs_backwards = gen_varrs_with(limit, &sample_backwards);

        for (varr_forwards, varr_backwards) in varrs_forwards.iter().zip(varrs_backwards.iter()) {
            let curr_forwards_len = varr_forwards.len();
            let curr_backwards_len = varr_backwards.len();

            let added = varr_forwards.add(varr_backwards).unwrap();
            let reference = sample_forwards.iter().zip(sample_backwards.iter()).map(|(x, y)| x + y);

            assert_eq!(added.len(), curr_forwards_len);
            assert_eq!(added.len(), curr_backwards_len);

            assert!(added.iter().zip(reference).all(|(x, y)| x == y));
        }
    }

    #[test]
    pub fn sub() {
        let limit = 32usize;
        let sample_forwards = (0..limit*2).step_by(2).collect::<Box<[Unit]>>();
        let sample_backwards = (0..limit).collect::<Box<[Unit]>>();

        let varrs_forwards = gen_varrs_with(limit, &sample_forwards);
        let varrs_backwards = gen_varrs_with(limit, &sample_backwards);

        for (varr_forwards, varr_backwards) in varrs_forwards.iter().zip(varrs_backwards.iter()) {
            let curr_forwards_len = varr_forwards.len();
            let curr_backwards_len = varr_backwards.len();

            let subbed = varr_forwards.sub(varr_backwards).unwrap();
            let reference = sample_forwards.iter().zip(sample_backwards.iter()).map(|(x, y)| x - y);

            assert_eq!(subbed.len(), curr_forwards_len);
            assert_eq!(subbed.len(), curr_backwards_len);

            assert!(subbed.iter().zip(reference).all(|(x, y)| x == y));
        }
    }


    #[test]
    pub fn div() {
        let limit = 32usize;
        let sample_forwards = (0..limit*512).step_by(512).collect::<Box<[Unit]>>();
        let sample_backwards = (1..=limit).collect::<Box<[Unit]>>();

        let varrs_forwards = gen_varrs_with(limit, &sample_forwards);
        let varrs_backwards = gen_varrs_with(limit, &sample_backwards);

        for (varr_forwards, varr_backwards) in varrs_forwards.iter().zip(varrs_backwards.iter()) {
            let curr_forwards_len = varr_forwards.len();
            let curr_backwards_len = varr_backwards.len();

            let divided = varr_forwards.div(varr_backwards).unwrap();
            let reference = sample_forwards.iter().zip(sample_backwards.iter()).map(|(x, y)| x / y);

            assert_eq!(divided.len(), curr_forwards_len);
            assert_eq!(divided.len(), curr_backwards_len);

            assert!(divided.iter().zip(reference).all(|(x, y)| x == y));
        }
    }

    #[test]
    pub fn mul() {
        let limit = 32usize;
        let sample_forwards = (0..limit*2).step_by(2).collect::<Box<[Unit]>>();
        let sample_backwards = (0..limit).collect::<Box<[Unit]>>();

        let varrs_forwards = gen_varrs_with(limit, &sample_forwards);
        let varrs_backwards = gen_varrs_with(limit, &sample_backwards);

        for (varr_forwards, varr_backwards) in varrs_forwards.iter().zip(varrs_backwards.iter()) {
            let curr_forwards_len = varr_forwards.len();
            let curr_backwards_len = varr_backwards.len();

            let multiplied = varr_forwards.mul(varr_backwards).unwrap();
            let reference = sample_forwards.iter().zip(sample_backwards.iter()).map(|(x, y)| x * y);

            assert_eq!(multiplied.len(), curr_forwards_len);
            assert_eq!(multiplied.len(), curr_backwards_len);

            assert!(multiplied.iter().zip(reference).all(|(x, y)| x == y));
        }
    }

    #[test]
    pub fn rem() {
        let limit = 32usize;
        let sample_forwards = (1..=limit).collect::<Box<[Unit]>>();
        let sample_backwards = (2..=limit*2).step_by(2).collect::<Box<[Unit]>>();

        let varrs_forwards = gen_varrs_with(limit, &sample_forwards);
        let varrs_backwards = gen_varrs_with(limit, &sample_backwards);

        for (varr_forwards, varr_backwards) in varrs_forwards.iter().zip(varrs_backwards.iter()) {
            let curr_forwards_len = varr_forwards.len();
            let curr_backwards_len = varr_backwards.len();

            let remainders = varr_forwards.rem(varr_backwards).unwrap();
            let reference = sample_forwards.iter().zip(sample_backwards.iter()).map(|(x, y)| x % y);

            assert_eq!(remainders.len(), curr_forwards_len);
            assert_eq!(remainders.len(), curr_backwards_len);

            assert!(remainders.iter().zip(reference).all(|(x, y)| x == y));
        }
    }
}