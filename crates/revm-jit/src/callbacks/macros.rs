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

macro_rules! resize_memory {
    ($ecx:expr, $offset:expr, $len:expr) => {
        match resize_memory($ecx, $offset, $len) {
            InstructionResult::Continue => {}
            ir => return ir,
        }
    };
}

/// Same as `read_words_rev`, but returns the arguments in the order they were passed.
macro_rules! read_words {
    ($sp:expr, $($words:ident),+ $(,)?) => {
        let reverse_tokens!($($words),+) = unsafe { read_words_rev($sp) };
    };
}

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
macro_rules! reverse_tokens {
	(@rev [$first:tt$(, $rest:tt)*] [$($rev:tt),*]) => {
		reverse_tokens! {
			@rev [$($rest),*][$first $(, $rev)*]
		}
	};
	(@rev [] [$($rev:tt),*]) => {
		[$($rev)*]
	};
	($($tt:tt)+) => {
		reverse_tokens! {
			@rev [$($tt),+] []
		}
	};
}

macro_rules! callbacks {
    (@count) => { 0 };
    (@count $first:tt $(, $rest:tt)*) => { 1 + callbacks!(@count $($rest),*) };

    (|$bcx:ident| { $($init:tt)* }
     $($ident:ident = $(#[$attr:expr])* $name:ident($($params:expr),* $(,)?) $ret:expr),* $(,)?
    ) => {
        /// Callbacks that can be called by the JIT-compiled functions.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub(crate) enum Callback {
            $($ident,)*
        }

        #[allow(unused_variables)]
        impl Callback {
            pub(crate) const COUNT: usize = callbacks!(@count $($ident),*);

            pub(crate) const fn name(self) -> &'static str {
                match self {
                    $(Self::$ident => stringify!($name),)*
                }
            }

            pub(crate) fn addr(self) -> usize {
                match self {
                    $(Self::$ident => self::$name as usize,)*
                }
            }

            pub(crate) fn ret<B: TypeMethods>(self, $bcx: &mut B) -> Option<B::Type> {
                $($init)*
                match self {
                    $(Self::$ident => $ret,)*
                }
            }

            pub(crate) fn params<B: TypeMethods>(self, $bcx: &mut B) -> Vec<B::Type> {
                $($init)*
                match self {
                    $(Self::$ident => vec![$($params),*],)*
                }
            }

            pub(crate) fn attrs(self) -> &'static [Attribute] {
                #[allow(unused_imports)]
                use Attribute::*;
                match self {
                    $(Self::$ident => &[$($attr)*]),*
                }
            }
        }
    };
}
