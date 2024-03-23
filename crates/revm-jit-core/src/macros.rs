/// Log the time it takes to execute the given expression.
#[macro_export]
macro_rules! time {
    ($level:expr, $what:literal, || $e:expr) => {{
        let res;
        let mut e = || $e;
        if $crate::private::tracing::enabled!($level) {
            let timer = std::time::Instant::now();
            res = e();
            // $crate::private::tracing::event!($level, elapsed=?timer.elapsed(), $what);
            $crate::private::tracing::event!($level, "{:<16} {:?}", $what, timer.elapsed());
        } else {
            res = e();
        }
        res
    }};
}

/// Log the time it takes to execute the given expression at `debug` level.
#[macro_export]
macro_rules! debug_time {
    ($what:literal, || $e:expr) => {
        $crate::time!($crate::private::tracing::Level::DEBUG, $what, || $e)
    };
}

/// Log the time it takes to execute the given expression at `trace` level.
#[macro_export]
macro_rules! trace_time {
    ($what:literal, || $e:expr) => {
        $crate::time!($crate::private::tracing::Level::TRACE, $what, || $e)
    };
}
