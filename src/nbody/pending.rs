//! Frames whose scattering is still being computed, e.g. on a GPU that
//! answers through the browser's event loop, where blocking would deadlock.

use anyhow::Result;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

type Job<T> = Pin<Box<dyn Future<Output = Result<T>>>>;

pub(super) struct PendingFrames<T> {
    jobs: Vec<(usize, Vec<f64>, Job<T>)>,
}

impl<T> PendingFrames<T> {
    pub fn new() -> Self {
        Self { jobs: Vec::new() }
    }

    /// `centres` travels with the frame so it is folded in with its beads.
    pub fn push(
        &mut self,
        copy: usize,
        centres: Vec<f64>,
        job: impl Future<Output = Result<T>> + 'static,
    ) {
        self.jobs.push((copy, centres, Box::pin(job)));
    }

    /// Finished frames in submission order. Polls once each, never blocks.
    /// Nothing waits on a wakeup: the browser's event loop lands GPU
    /// readbacks by itself, and native pripps polls the device before its
    /// first await, so a no-op waker suffices and the next call polls again.
    pub fn drain_ready(&mut self) -> Vec<Result<(usize, Vec<f64>, T)>> {
        let mut context = Context::from_waker(Waker::noop());
        let mut ready = Vec::new();
        self.jobs.retain_mut(|(copy, centres, job)| {
            let Poll::Ready(result) = job.as_mut().poll(&mut context) else {
                return true;
            };
            let centres = std::mem::take(centres);
            ready.push(result.map(|value| (*copy, centres, value)));
            false
        });
        ready
    }

    /// Drops unfinished frames; their results are discarded on arrival.
    pub fn clear(&mut self) {
        self.jobs.clear();
    }

    pub fn len(&self) -> usize {
        self.jobs.len()
    }
}

/// Result of a future that must complete on its first poll, such as a CPU
/// sum; unlike `block_on` this can't deadlock a browser if it doesn't.
pub(super) fn ready_now<T, E: Into<anyhow::Error>>(
    job: impl Future<Output = std::result::Result<T, E>>,
) -> Result<T> {
    let mut context = Context::from_waker(Waker::noop());
    match std::pin::pin!(job).poll(&mut context) {
        Poll::Ready(result) => result.map_err(Into::into),
        Poll::Pending => anyhow::bail!("a synchronous sum did not complete"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    /// Pending until `flag` is set, like a GPU readback.
    fn gated(flag: &Rc<Cell<bool>>, value: i32) -> impl Future<Output = Result<i32>> {
        let flag = flag.clone();
        std::future::poll_fn(move |_| match flag.get() {
            true => Poll::Ready(Ok(value)),
            false => Poll::Pending,
        })
    }

    fn drained(pending: &mut PendingFrames<i32>) -> Vec<(usize, Vec<f64>, i32)> {
        pending
            .drain_ready()
            .into_iter()
            .map(|r| r.unwrap())
            .collect()
    }

    #[test]
    fn ready_frames_drain_at_once() {
        let mut pending = PendingFrames::new();
        pending.push(3, vec![1.0], async { Ok(7) });
        assert_eq!(drained(&mut pending), [(3, vec![1.0], 7)]);
        assert_eq!(pending.len(), 0);
    }

    #[test]
    fn unfinished_frames_wait_for_their_result() {
        let flag = Rc::new(Cell::new(false));
        let mut pending = PendingFrames::new();
        pending.push(0, vec![], gated(&flag, 1));
        assert!(drained(&mut pending).is_empty());
        assert_eq!(pending.len(), 1);
        flag.set(true);
        assert_eq!(drained(&mut pending), [(0, vec![], 1)]);
    }

    #[test]
    fn cleared_frames_never_arrive() {
        let flag = Rc::new(Cell::new(false));
        let mut pending = PendingFrames::new();
        pending.push(0, vec![], gated(&flag, 1));
        pending.clear();
        flag.set(true);
        assert!(drained(&mut pending).is_empty());
    }

    #[test]
    fn failed_frames_come_out_as_errors() {
        let mut pending = PendingFrames::<i32>::new();
        pending.push(0, vec![], async { Err(anyhow::anyhow!("lost device")) });
        let drained = pending.drain_ready();
        assert_eq!(drained.len(), 1);
        let error = drained.into_iter().next().unwrap().unwrap_err();
        assert!(error.to_string().contains("lost device"));
    }
}
