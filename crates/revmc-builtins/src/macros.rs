#[allow(unused_macros)]
#[collapse_debuginfo(yes)]
macro_rules! tri {
    ($e:expr) => {
        match $e {
            Ok(x) => x,
            Err(_) => return Err(BuiltinError::from(InstructionResult::InvalidOperandOOG)),
        }
    };
}

#[collapse_debuginfo(yes)]
macro_rules! gas {
    ($ecx:expr, $gas:expr) => {
        if !$ecx.gas.record_cost($gas) {
            return Err(BuiltinError::from(InstructionResult::OutOfGas));
        }
    };
}

#[collapse_debuginfo(yes)]
macro_rules! gas_opt {
    ($ecx:expr, $gas:expr) => {
        match $gas {
            Some(gas) => gas!($ecx, gas),
            None => return Err(BuiltinError::from(InstructionResult::OutOfGas)),
        }
    };
}

/// Mirrors revm's `berlin_load_account!` macro.
/// Loads account info with the cold-load-skip optimization: if remaining gas
/// is less than the cold cost, skip the DB load and return OOG immediately.
/// Charges `cold_account_additional_cost` if cold. The base `warm_storage_read_cost`
/// is deducted upfront by the JIT as static gas.
#[collapse_debuginfo(yes)]
macro_rules! berlin_load_account {
    ($ecx:expr, $address:expr, $load_code:expr) => {{
        let cold_load_gas = $ecx.host.gas_params().cold_account_additional_cost();
        let skip_cold_load = $ecx.gas.remaining() < cold_load_gas;
        let account =
            $ecx.host.load_account_info_skip_cold_load($address, $load_code, skip_cold_load)?;
        if account.is_cold {
            gas!($ecx, cold_load_gas);
        }
        account
    }};
}

#[collapse_debuginfo(yes)]
macro_rules! ensure_non_staticcall {
    ($ecx:expr) => {
        if $ecx.is_static {
            return Err(BuiltinError::from(InstructionResult::StateChangeDuringStaticCall));
        }
    };
}

#[collapse_debuginfo(yes)]
macro_rules! ensure_memory {
    ($ecx:expr, $offset:expr, $len:expr) => {
        ensure_memory($ecx, $offset, $len)?
    };
}

/// Same as `read_words_rev`, but returns the arguments in the order they were passed.
#[collapse_debuginfo(yes)]
macro_rules! read_words {
    ($sp:expr, $($words:ident),+ $(,)?) => {
        let rev![$($words),+] = unsafe { read_words_rev($sp) };
    };
}

#[collapse_debuginfo(yes)]
macro_rules! pop {
    ($sp:expr; $($x:ident),* $(,)?) => {
        $(
            $sp = $sp.sub(1);
            let $x = &mut *$sp;
        )*
    };
}

#[collapse_debuginfo(yes)]
macro_rules! try_into_usize {
    ($x:expr) => {
        match $x.to_u256().as_limbs() {
            x => {
                if (x[0] > usize::MAX as u64) | (x[1] != 0) | (x[2] != 0) | (x[3] != 0) {
                    return Err(BuiltinError::from(InstructionResult::InvalidOperandOOG));
                }
                x[0] as usize
            }
        }
    };
}

// Credits: <https://github.com/AuroransSolis/rustconf-2023/blob/665a645d751dfe0e483261e3abca25ab4bb9e13a/reverse-tokens/src/main.rs>
#[collapse_debuginfo(yes)]
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

#[collapse_debuginfo(yes)]
macro_rules! debug_unreachable {
    ($($t:tt)*) => {
        if cfg!(debug_assertions) {
            unreachable!($($t)*);
        } else {
            unsafe { core::hint::unreachable_unchecked() };
        }
    };
}

#[collapse_debuginfo(yes)]
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
