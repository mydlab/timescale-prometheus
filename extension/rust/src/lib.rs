
use std::{
    collections::VecDeque,
    mem::size_of,
    ptr::null_mut,
};

use ts_extend::{
    elog,
    elog::Level::Error,
    palloc::memory_context_switch_to,
    pg_fn,
    pg_sys::{
        AggCheckCallContext,
        ArrayType,
        FLOAT8OID,
        FLOAT8PASSBYVAL,
        MemoryContext,
        TimestampTz,
        construct_md_array,
    },
};

type Seconds = i64;
const USECS_PER_SEC: i64 = 1_000_000;

pg_fn!{
    // prom divides time into no-sliding windows of fixed size, e.g.
    // |  5 seconds  |  5 seconds  |  5 seconds  |  5 seconds  |  5 seconds  |
    // we take the first and last values in that bucket and uses `last-first` as the
    // value for that bucket.
    //  | a b c d e | f g h i | j   k |   m    |
    //  |   e - a   |  i - f  | k - j | <null> |
    pub fn gapfill_delta_transition(
        state: Option<*mut GapfillDeltaTransition>,
        lowest_time: TimestampTz,
        greatest_time: TimestampTz,
        step_size: Seconds, // `prev_now - step` is where the next window starts
        window_size: Seconds, // the size of a window to delta over
        time: TimestampTz,
        val: f64;
        fcinfo
    ) -> Option<*mut GapfillDeltaTransition> {
        let mut agg_ctx: MemoryContext = null_mut();

        if unsafe {AggCheckCallContext(fcinfo, &mut agg_ctx) == 0} {
            elog!(Error, "must call gapfill_delta_transition as an aggregate")
        }

        if time <= lowest_time || time > greatest_time {
            elog!(Error, "input time less than lowest time")
        }

        let old_ctx = unsafe { memory_context_switch_to(agg_ctx) };

        let state = state.map(|s| unsafe { &mut *s }).unwrap_or_else(|| {
            let expected_deltas = ((greatest_time - lowest_time) / (step_size * USECS_PER_SEC)) + 1;
            let state = GapfillDeltaTransition{
                window: VecDeque::default(),
                deltas: Vec::with_capacity(expected_deltas as _),
                nulls: Vec::with_capacity(expected_deltas as _),
                current_window_max: greatest_time,
                current_window_min: greatest_time - window_size*USECS_PER_SEC,
                step_size: step_size*USECS_PER_SEC,
            }.into();
            Box::leak(state)
        });
        state.add_data_point(time, val);

        unsafe { memory_context_switch_to(old_ctx); }

        Some(state as *mut _)
    }
}

pg_fn!{
    pub fn gapfill_delta_final(
        state: Option<*mut GapfillDeltaTransition>;
        fcinfo
    ) -> Option<*mut ArrayType> {
        let mut agg_ctx: MemoryContext = null_mut();

        if unsafe {AggCheckCallContext(fcinfo, &mut agg_ctx) == 0} {
            elog!(Error, "must call gapfill_delta_transition as an aggregate")
        }

        let old_ctx = unsafe { memory_context_switch_to(agg_ctx) };

        let ret = state.map(|s| unsafe { (&mut *s).to_pg_array() });

        unsafe { memory_context_switch_to(old_ctx); }

        ret
    }
}

struct GapfillDeltaTransition {
    window: VecDeque<(TimestampTz, f64)>,
    deltas: Vec<f64>,
    nulls: Vec<bool>,
    current_window_max: TimestampTz,
    current_window_min: TimestampTz,
    step_size: TimestampTz,
}

impl GapfillDeltaTransition {
    pub fn add_data_point(&mut self, time: TimestampTz, val: f64) {
        while !self.in_current_window(time) {
            self.flush_current_window()
        }

        self.window.push_back((time, val))
    }

    fn in_current_window(&self, time: TimestampTz) -> bool {
        time > self.current_window_min
    }

    fn flush_current_window(&mut self) {
        match (self.window.front(), self.window.back()) {
            (Some((_, latest_val)), Some((_, earliest_val))) => {
                self.deltas.push(latest_val - earliest_val);
                self.nulls.push(false);
            },
            // if there are 1 or fewer values in the window, store NULL
            (_, _) => self.nulls.push(true),
        }

        self.current_window_min -= self.step_size;
        self.current_window_max -= self.step_size;

        let current_window_max = self.current_window_max;
        self.window.drain(..)
            .take_while(|(time, _)| *time > current_window_max)
            .for_each(|_|())
    }

    pub fn to_pg_array(&mut self) -> *mut ArrayType{
        self.flush_current_window();
        unsafe {
            construct_md_array(
                self.deltas.as_mut_ptr() as *mut _, //TODO create safe converter
                self.nulls.as_mut_ptr(),
                1,
                &mut (self.deltas.len() as _),
                &mut 1,
                FLOAT8OID,
                size_of::<f64>() as _,
                FLOAT8PASSBYVAL != 0,
                'd' as u8 as _,
            )
        }
    }
}

