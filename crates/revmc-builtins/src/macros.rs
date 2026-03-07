#[allow(unused_macros)]
macro_rules! tri {
    ($e:expr) => {
        match $e {
            Ok(x) => x,
            Err(_) => return InstructionResult::InvalidOperandOOG,
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
            InstructionResult::Stop => {}
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

/// Mirrors revm's `berlin_load_account!` macro.
/// Loads account info with the cold-load-skip optimization: if remaining gas
/// is less than the cold cost, skip the DB load and return OOG immediately.
/// Only charges `cold_account_additional_cost` if cold; the caller is responsible
/// for any static/warm gas.
macro_rules! berlin_load_account {
    ($ecx:expr, $address:expr, $load_code:expr) => {{
        use revm_context_interface::host::LoadError;
        let cold_load_gas = $ecx.host.gas_params().cold_account_additional_cost();
        let skip_cold_load = $ecx.gas.remaining() < cold_load_gas;
        match $ecx.host.load_account_info_skip_cold_load($address, $load_code, skip_cold_load) {
            Ok(account) => {
                if account.is_cold {
                    gas!($ecx, cold_load_gas);
                }
                account
            }
            Err(LoadError::ColdLoadSkipped) => return InstructionResult::OutOfGas,
            Err(LoadError::DBError) => return InstructionResult::FatalExternalError,
        }
    }};
}

/// Mirrors revm's SLOAD cold-load-skip pattern.
macro_rules! berlin_sload {
    ($ecx:expr, $address:expr, $key:expr) => {{
        use revm_context_interface::host::LoadError;
        let additional_cold_cost = $ecx.host.gas_params().cold_storage_additional_cost();
        let skip_cold = $ecx.gas.remaining() < additional_cold_cost;
        match $ecx.host.sload_skip_cold_load($address, $key, skip_cold) {
            Ok(storage) => {
                if storage.is_cold {
                    gas!($ecx, additional_cold_cost);
                }
                storage
            }
            Err(LoadError::ColdLoadSkipped) => return InstructionResult::OutOfGas,
            Err(LoadError::DBError) => return InstructionResult::FatalExternalError,
        }
    }};
}

/// Mirrors revm's SSTORE cold-load-skip pattern.
macro_rules! berlin_sstore {
    ($ecx:expr, $address:expr, $key:expr, $value:expr) => {{
        use revm_context_interface::host::LoadError;
        let additional_cold_cost = $ecx.host.gas_params().cold_storage_additional_cost();
        let skip_cold = $ecx.gas.remaining() < additional_cold_cost;
        match $ecx.host.sstore_skip_cold_load($address, $key, $value, skip_cold) {
            Ok(load) => load,
            Err(LoadError::ColdLoadSkipped) => return InstructionResult::OutOfGas,
            Err(LoadError::DBError) => return InstructionResult::FatalExternalError,
        }
    }};
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
