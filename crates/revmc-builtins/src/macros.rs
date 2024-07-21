#[allow(unused_macros)]
macro_rules! tri {
    ($e:expr) => {
        match $e {
            Ok(x) => x,
            Err(_) => return InstructionResult::InvalidOperandOOG,
        }
    };
}

macro_rules! try_opt {
    ($e:expr) => {
        match $e {
            Some(x) => x,
            None => return InstructionResult::InvalidOperandOOG,
        }
    };
}

macro_rules! try_host {
    ($e:expr) => {
        match $e {
            Some(x) => x,
            None => return InstructionResult::FatalExternalError,
        }
    };
}

macro_rules! try_ir {
    ($e:expr) => {
        match $e {
            InstructionResult::Continue => {}
            ir => return ir,
        }
    };
}

macro_rules! gas {
    ($ecx:expr, $gas:expr) => {
        if !$ecx.gas.record_cost($gas) {
            return InstructionResult::OutOfGas;
        }
    };
}

macro_rules! gas_opt {
    ($ecx:expr, $gas:expr) => {
        match $gas {
            Some(gas) => gas!($ecx, gas),
            None => return InstructionResult::OutOfGas,
        }
    };
}

macro_rules! ensure_non_staticcall {
    ($ecx:expr) => {
        if $ecx.is_static {
            return InstructionResult::StateChangeDuringStaticCall;
        }
    };
}

macro_rules! ensure_memory {
    ($ecx:expr, $offset:expr, $len:expr) => {
        try_ir!(ensure_memory($ecx, $offset, $len))
    };
}

/// Same as `read_words_rev`, but returns the arguments in the order they were passed.
macro_rules! read_words {
    ($sp:expr, $($words:ident),+ $(,)?) => {
        let rev![$($words),+] = unsafe { read_words_rev($sp) };
    };
}

macro_rules! pop {
    ($sp:expr; $($x:ident),* $(,)?) => {
        $(
            $sp = $sp.sub(1);
            let $x = &mut *$sp;
        )*
    };
}

macro_rules! try_into_usize {
    ($x:expr) => {
        match $x.to_u256().as_limbs() {
            x => {
                if (x[0] > usize::MAX as u64) | (x[1] != 0) | (x[2] != 0) | (x[3] != 0) {
                    return InstructionResult::InvalidOperandOOG;
                }
                x[0] as usize
            }
        }
    };
}

// Credits: <https://github.com/AuroransSolis/rustconf-2023/blob/665a645d751dfe0e483261e3abca25ab4bb9e13a/reverse-tokens/src/main.rs>
macro_rules! rev {
	(@rev [$first:tt$(, $rest:tt)*] [$($rev:tt),*]) => {
		rev! {
			@rev [$($rest),*][$first $(, $rev)*]
		}
	};
	(@rev [] [$($rev:tt),*]) => {
		[$($rev)*] // NOTE: Extra `[]` to make this an array pattern.
	};
	($($tt:tt)+) => {
		rev! {
			@rev [$($tt),+] []
		}
	};
}

macro_rules! debug_unreachable {
    ($($t:tt)*) => {
        if cfg!(debug_assertions) {
            unreachable!($($t)*);
        } else {
            unsafe { core::hint::unreachable_unchecked() };
        }
    };
}

macro_rules! assume {
    ($e:expr $(,)?) => {
        if !$e {
            debug_unreachable!(stringify!($e));
        }
    };

    ($e:expr, $($t:tt)+) => {
        if !$e {
            debug_unreachable!($($t)+);
        }
    };
}
