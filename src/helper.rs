use std::{sync::MutexGuard, collections::HashMap, ops::{Deref, DerefMut}, hash::Hash, fmt::Display};
use ndarray::ArrayViewD;

use crate::operations::Tensor;

/// This structure, if dereferenced, returns the value identified by the given key, referencing a [`HashMap`] locked behind a
/// Mutex.
/// 
/// # Example
/// ```ignore
/// let mut map = Mutex::new(HashMap::from([
///     ("foo", "Hello World!"),
///     ("bar", "!dlroW olleH")
/// ]));
/// let foo_guard = InnerMutexGuard(map.lock().unwrap(), "foo");
/// assert_eq!(*foo_guard, "Hello World!");
/// ```
/// 
/// # Panics
/// If the map has no value associated to the key.
pub(crate) struct InnerMutexGuard<'a, K, V>(pub MutexGuard<'a, HashMap<K, V>>, pub K);

impl<'a, K: Hash + Eq + PartialEq, V> Deref for InnerMutexGuard<'a, K,V> {
    type Target = V;
    fn deref(&self) -> &Self::Target {
        let Self(guard, k) = self;
        guard.get(k).unwrap()
    }
}

impl<'a, K: Hash + Eq + PartialEq, V> DerefMut for InnerMutexGuard<'a, K,V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let Self(guard, k) = self;
        guard.get_mut(k).unwrap()
    }
}

/// Wrapper structure used to pretty-print a multidimensional array (tensor)
pub struct PrettyTensor<'a> {
    tensor: &'a Tensor
}

impl<'a> From<&'a Tensor> for PrettyTensor<'a> {
    fn from(tensor: &'a Tensor) -> Self {
        Self { tensor }
    }
}

impl<'a> PrettyTensor<'a> {
    /// Recursively prints the contents of a multidimensional array.
    /// 
    /// Each element takes up one line.
    fn print_rec(tensor: &ArrayViewD<f32>, f: &mut std::fmt::Formatter, level: usize) -> std::fmt::Result {
        writeln!(f, "{}[", " ".repeat(level*2))?;
        let res =
            tensor
                .outer_iter()
                .map(|slice| {
                    if slice.ndim() > 1 {
                        Self::print_rec(&slice, f, level+1)?;
                    } else {
                        for v in slice {
                            writeln!(f, "{}{v},", " ".repeat((level+1)*2))?;
                        }
                    }
                    Ok(())
                })
                .collect();
        writeln!(f, "{}],", " ".repeat(level*2))?;
        res
    }
}

impl<'a> Display for PrettyTensor<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::print_rec(&self.tensor.view(), f, 0)
    }
}