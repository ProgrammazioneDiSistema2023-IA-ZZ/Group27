use std::{sync::MutexGuard, collections::HashMap, ops::{Deref, DerefMut}, hash::Hash};

/// La struttura, se dereferenziata, restituisce il valore identificato dalla chiave data, con riferimento ad una [`HashMap`]
/// controllata da un [`Mutex`][`std::sync::Mutex`].
/// 
/// # Esempio
/// ```ignore
/// let mut map = Mutex::new(HashMap::from([
///     ("foo", "Hello World!"),
///     ("bar", "!dlroW olleH")
/// ]));
/// let foo_guard = InnerMutexGuard(map.lock().unwrap(), "foo")
/// assert_eq!(*foo_guard, "Hello World!");
/// ```
/// 
/// # Panico
/// Se la chiave non è associata ad alcun valore nella HashMap.
pub struct InnerMutexGuard<'a, K, V>(pub MutexGuard<'a, HashMap<K, V>>, pub K);

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