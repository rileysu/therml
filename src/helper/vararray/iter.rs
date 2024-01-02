use super::{VarArray, Unit};

pub struct Iter<'a> {
    varr: &'a VarArray,
    pos: usize,
    is_done: bool,
}

impl<'a> Iter<'a> {
    pub fn new(varr: &'a VarArray) -> Self {
        Self {
            varr,
            pos: 0,
            is_done: false,
        }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = Unit;

    fn next(&mut self) -> Option<Self::Item> {
        //It is faster to match again rather than using get since we don't need to check bounds
        if self.is_done {
            None
        } else { 
            let out = match self.varr {
                VarArray::Zero(_) => None,
                VarArray::One(arr) => match self.pos {
                    0 => Some(arr[0]),
                    _ => {
                        self.is_done = true;
                        None
                    },
                },
                VarArray::Two(arr) => match self.pos {
                    0 => Some(arr[0]),
                    1 => Some(arr[1]),
                    _ => {
                        self.is_done = true;
                        None
                    },
                },
                VarArray::Three(arr) => match self.pos {
                    0 => Some(arr[0]),
                    1 => Some(arr[1]),
                    2 => Some(arr[2]),
                    _ => {
                        self.is_done = true;
                        None
                    },
                },
                VarArray::Four(arr) => match self.pos {
                    0 => Some(arr[0]),
                    1 => Some(arr[1]),
                    2 => Some(arr[2]),
                    3 => Some(arr[3]),
                    _ => {
                        self.is_done = true;
                        None
                    },
                },
                VarArray::Five(arr) => match self.pos {
                    0 => Some(arr[0]),
                    1 => Some(arr[1]),
                    2 => Some(arr[2]),
                    3 => Some(arr[3]),
                    4 => Some(arr[4]),
                    _ => {
                        self.is_done = true;
                        None
                    },
                },
                VarArray::Six(arr) => match self.pos {
                    0 => Some(arr[0]),
                    1 => Some(arr[1]),
                    2 => Some(arr[2]),
                    3 => Some(arr[3]),
                    4 => Some(arr[4]),
                    5 => Some(arr[5]),
                    _ => {
                        self.is_done = true;
                        None
                    },
                },
                VarArray::Seven(arr) => match self.pos {
                    0 => Some(arr[0]),
                    1 => Some(arr[1]),
                    2 => Some(arr[2]),
                    3 => Some(arr[3]),
                    4 => Some(arr[4]),
                    5 => Some(arr[5]),
                    6 => Some(arr[6]),
                    _ => {
                        self.is_done = true;
                        None
                    },
                },
                VarArray::Eight(arr) => match self.pos {
                    0 => Some(arr[0]),
                    1 => Some(arr[1]),
                    2 => Some(arr[2]),
                    3 => Some(arr[3]),
                    4 => Some(arr[4]),
                    5 => Some(arr[5]),
                    6 => Some(arr[6]),
                    7 => Some(arr[7]),
                    _ => {
                        self.is_done = true;
                        None
                    },
                },
                VarArray::Etc(arr) => arr.get(self.pos).copied(),
            };

            if out.is_some() {
                self.pos += 1
            }

            out
        }
    }
}