/// Log the time it takes to execute the given expression.
#[macro_export]
macro_rules! time {
    ($level:expr, $what:literal, || $e:expr) => {
        if $crate::private::tracing::enabled!($level) {
            let timer = std::time::Instant::now();
            let res = $e;
            // $crate::private::tracing::event!($level, elapsed=?timer.elapsed(), $what);
            $crate::private::tracing::event!($level, "{:<30} {:?}", $what, timer.elapsed());
            res
        } else {
            $e
        }
    };
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
