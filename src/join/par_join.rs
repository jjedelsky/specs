
use std::marker::PhantomData;
use std::cell::UnsafeCell;

use hibitset::{BitProducer, BitSetLike};

use rayon::iter::plumbing::{bridge_unindexed, Folder, UnindexedConsumer, UnindexedProducer};
use rayon::iter::ParallelIterator;
use join::Join;


/// The purpose of the `ParJoin` trait is to provide a way
/// to access multiple storages in parallel at the same time with
/// the merged bit set.
pub unsafe trait ParJoin<'a>: Join<'a> {
    /// Create a joined parallel iterator over the contents.
    fn par_join(self) -> JoinParIter<'a, Self>
    where
        Self: Sized,
    {
        if <Self as Join>::is_unconstrained() {
            println!("WARNING: `ParJoin` possibly iterating through all indices, you might've made a join with all `MaybeJoin`s, which is unbounded in length.");
        }

        JoinParIter(self, PhantomData)
    }
}

/// `JoinParIter` is a `ParallelIterator` over a group of `Storages`.
#[must_use]
pub struct JoinParIter<'a, J: Join<'a>>(J, PhantomData<&'a mut ()>);

impl<'a, J> ParallelIterator for JoinParIter<'a, J>
where
    J: 'a + Join<'a> + Send,
    J::Mask: Send + Sync,
    J::Type: Send,
    J::Value: Send,
{
    type Item = J::Type;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        let (keys, values) = unsafe { self.0.open() };
        // Create a bit producer which splits on up to three levels
        let producer = BitProducer((&keys).iter(), 3);
        let values = UnsafeCell::new(values);

        bridge_unindexed(JoinProducer::<J>::new(producer, &values), consumer)
    }
}

struct JoinProducer<'a, 'b, J>
where
    J: Join<'b> + Send,
    J::Mask: Send + Sync + 'a,
    J::Type: Send,
    J::Value: Send + 'a,
{
    keys: BitProducer<'a, <J as Join<'b>>::Mask>,
    values: &'a UnsafeCell<<J as Join<'b>>::Value>,
}

impl<'a, 'b: 'a, J> JoinProducer<'a, 'b, J>
where
    J: 'a + Join<'b> + Send,
    J::Type: Send,
    J::Value: 'a + Send,
    J::Mask: 'a + Send + Sync,
{
    fn new(keys: BitProducer<'a, <J as Join<'b>>::Mask>, values: &'a UnsafeCell<<J as Join<'b>>::Value>) -> Self {
        JoinProducer { keys, values }
    }
}

unsafe impl<'a, 'b: 'a, J> Send for JoinProducer<'a, 'b, J>
where
    J: 'a + Join<'b> + Send,
    J::Type: Send,
    J::Value: 'a + Send,
    J::Mask: 'a + Send + Sync,
{}

impl<'a, 'b: 'a, J> UnindexedProducer for JoinProducer<'a, 'b, J>
where
    J: 'a + Join<'b> + Send,
    J::Type: Send,
    J::Value: 'a + Send,
    J::Mask: 'a + Send + Sync,
{
    type Item = J::Type;
    fn split(self) -> (Self, Option<Self>) {
        let (cur, other) = self.keys.split();
        let values = self.values;
        let first = JoinProducer::new(cur, values);
        let second = other.map(|o| JoinProducer::new(o, values));

        (first, second)
    }

    fn fold_with<F>(self, folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        let JoinProducer { values, keys, .. } = self;
        let iter = keys.0.map(|idx| unsafe {
            // This unsafe block should be safe if the `J::get`
            // can be safely called from different threads with distinct indices.

            // The indices here are guaranteed to be distinct because of the fact
            // that the bit set is split.
            J::get(&mut *values.get(), idx)
        });

        folder.consume_iter(iter)
    }
}