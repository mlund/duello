use anyhow::Result;
use get_size::GetSize;

/// Periodic table that emulate periodicity by padding edges
#[derive(Debug, Clone, GetSize)]
pub struct PaddedTable<T: Clone> {
    /// Minimum key value
    min: f64,
    /// Maximum key value
    _max: f64,
    /// Key resolution
    res: f64,
    /// Data of the table
    data: Vec<T>,
}

impl<T: Clone> PaddedTable<T> {
    pub fn new(min: f64, max: f64, step: f64, initial_value: T) -> PaddedTable<T> {
        assert!(min < max && step > 0.0);
        let n = ((max - min + 2.0 * step) / step + 0.5) as usize;
        Self {
            min: min - step,
            _max: max + step,
            res: step,
            data: vec![initial_value; n],
        }
    }

    /// Get minimum key value (inclusive)
    pub fn min_key(&self) -> f64 {
        self.min + self.res
    }

    /// Get maximum key value (inclusive)
    pub fn max_key(&self) -> f64 {
        self._max - self.res
    }

    /// Get key spacing
    pub fn key_step(&self) -> f64 {
        self.res
    }

    /// Convert value to index
    pub fn to_index(&self, key: f64) -> Result<usize> {
        let index = ((key - self.min) / self.res + 0.5) as usize;
        if index >= self.data.len() {
            anyhow::bail!("Index out of range")
        } else {
            Ok(index)
        }
    }
    /// Set value corresponding to a key and add periodic padding to first and last element
    ///
    /// Adds two extra padding elements to both ends to mimic periodicity:
    ///
    /// ```console
    /// ● | ○ ... ●   | ○
    /// 0 | 1 ... n-2 | n-1
    /// ```
    pub fn set(&mut self, key: f64, value: T) -> Result<()> {
        let n = self.data.len();
        let index = self.to_index(key)?;

        if index == 0 || index == n - 1 {
            // anyhow::bail!("Cannot set value in padded region")
        } else if index == 1 {
            self.data[n - 1] = value.clone();
        } else if index == n - 2 {
            self.data[0] = value.clone();
        }

        self.data[index] = value;
        Ok(())
    }

    pub fn get(&self, key: f64) -> Result<&T> {
        let index = self.to_index(key)?;
        Ok(&self.data[index])
    }

    pub fn get_mut(&mut self, key: f64) -> Result<&mut T> {
        let index = self.to_index(key)?;
        Ok(&mut self.data[index])
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

pub type PaddedTable1D = PaddedTable<f64>;
pub type PaddedTable2D = PaddedTable<PaddedTable1D>;

#[cfg(test)]
mod tests {
    use super::*;
    use iter_num_tools::arange;
    use std::f64::consts::PI;

    #[test]
    fn test_table() {
        let dx = 0.1;
        let mut table = PaddedTable::new(0.0, 1.0, dx, 0.0);
        let n = table.len();
        assert_eq!(n, 12);
        table.set(0.0, 1.0).unwrap();
        table.set(0.1, 2.0).unwrap();
        table.set(0.2, 3.0).unwrap();
        table.set(0.3, 4.0).unwrap();
        table.set(0.4, 5.0).unwrap();
        table.set(0.5, 6.0).unwrap();
        table.set(0.6, 7.0).unwrap();
        table.set(0.7, 8.0).unwrap();
        table.set(0.8, 9.0).unwrap();
        table.set(0.9, 10.0).unwrap();
        assert_eq!(*table.get(0.0).unwrap(), 1.0);
        assert_eq!(*table.get(0.1).unwrap(), 2.0);
        assert_eq!(*table.get(0.2).unwrap(), 3.0);
        assert_eq!(*table.get(0.3).unwrap(), 4.0);
        assert_eq!(*table.get(0.4).unwrap(), 5.0);
        assert_eq!(*table.get(0.5).unwrap(), 6.0);
        assert_eq!(*table.get(0.6).unwrap(), 7.0);
        assert_eq!(*table.get(0.7).unwrap(), 8.0);
        assert_eq!(*table.get(0.8).unwrap(), 9.0);
        assert_eq!(*table.get(0.9).unwrap(), 10.0);

        // error if direct insertion in padded regions
        // assert!(table.set(0.0 - dx, 0.0).is_err());
        // assert!(table.set(1.0, 0.0).is_err());

        // check lower padding
        assert_eq!(table.to_index(0.0 - dx).unwrap(), 0);
        assert_eq!(*table.get(0.0 - dx).unwrap(), 10.0);

        // check upper padding
        assert_eq!(table.to_index(1.0).unwrap(), n - 1);
        assert_eq!(*table.get(1.0).unwrap(), 1.0);
    }

    #[test]
    fn test_table_angles() {
        let res = 0.1;
        let angles = arange(0.0..2.0 * PI, res).collect::<Vec<f64>>();

        let mut table = PaddedTable::<usize>::new(0.0, 2.0 * PI, res, 0);
        assert_eq!(table.len(), angles.len() + 2);

        for (i, angle) in angles.iter().enumerate() {
            table.set(*angle, i).unwrap();
        }
        assert_eq!(*table.get(0.0).unwrap(), 0);
        assert_eq!(*table.get(2.0 * PI - res).unwrap(), angles.len() - 1);

        // access upper padding -> first angle
        assert_eq!(*table.get(2.0 * PI).unwrap(), 0);

        // access lower padding -> last angle
        assert_eq!(*table.get(-res).unwrap(), angles.len() - 1);
    }
}
