#[macro_export]
macro_rules! tri {
    ($e:expr) => {
        match $e {
            Ok(x) => x,
            Err(_) => return InstructionResult::InvalidOperandOOG,
        }
    };
}

#[macro_export]
macro_rules! try_opt {
    ($e:expr) => {
        match $e {
            Some(x) => x,
            None => return InstructionResult::InvalidOperandOOG,
        }
    };
}

#[macro_export]
macro_rules! try_host {
    ($e:expr) => {
        match $e {
            Some(x) => x,
            None => return InstructionResult::FatalExternalError,
        }
    };
}

#[macro_export]
macro_rules! gas {
    ($ecx:expr, $gas:expr) => {
        if !$ecx.gas.record_cost($gas) {
            return InstructionResult::OutOfGas;
        }
    };
}

#[macro_export]
macro_rules! gas_opt {
    ($ecx:expr, $gas:expr) => {
        match $gas {
            Some(gas) => gas!($ecx, gas),
            None => return InstructionResult::OutOfGas,
        }
    };
}

#[macro_export]
macro_rules! resize_memory {
    ($ecx:expr, $offset:expr, $len:expr) => {
        match resize_memory($ecx, $offset, $len) {
            InstructionResult::Continue => {}
            ir => return ir,
        }
    };
}

/// Same as `read_words_rev`, but returns the arguments in the order they were passed.
#[macro_export]
macro_rules! read_words {
    ($sp:expr, $($words:ident),+ $(,)?) => {
        let rev![$($words),+] = unsafe { read_words_rev($sp) };
    };
}

#[macro_export]
macro_rules! try_into_usize {
    ($x:expr) => {
        match $x {
            x => {
                let x = x.as_limbs();
                if x[1] != 0 || x[2] != 0 || x[3] != 0 {
                    return InstructionResult::InvalidOperandOOG;
                }
                let Ok(val) = usize::try_from(x[0]) else {
                    return InstructionResult::InvalidOperandOOG;
                };
                val
            }
        }
    };
}

// Credits: <https://github.com/AuroransSolis/rustconf-2023/blob/665a645d751dfe0e483261e3abca25ab4bb9e13a/reverse-tokens/src/main.rs>
#[macro_export]
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