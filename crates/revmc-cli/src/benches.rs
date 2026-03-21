use revm_bytecode::opcode as op;
use revmc::{U256, primitives::hex};
use std::hint::black_box;

macro_rules! include_code_str {
    ($path:literal) => {
        crate::read_code_string(
            include_bytes!($path),
            std::path::Path::new($path).extension().and_then(|s| s.to_str()),
        )
    };
}

#[derive(Clone, Debug)]
pub struct Bench {
    pub name: &'static str,
    pub kind: BenchKind,
}

#[derive(Clone, Debug)]
pub enum BenchKind {
    Bytecode { bytecode: Vec<u8>, calldata: Vec<u8>, stack_input: Vec<U256>, native: Option<fn()> },
    TxFixture(FixtureBenchDef),
}

/// Definition for a full-EVM transaction fixture benchmark.
#[derive(Clone, Debug)]
pub struct FixtureBenchDef {
    pub spec_id: revmc::primitives::hardfork::SpecId,
    pub fixture_json: &'static str,
}

impl Bench {
    /// Returns the bytecode fields, if this is a bytecode bench.
    pub fn as_bytecode(&self) -> Option<(&[u8], &[u8], &[U256], Option<fn()>)> {
        match &self.kind {
            BenchKind::Bytecode { bytecode, calldata, stack_input, native } => {
                Some((bytecode, calldata, stack_input, *native))
            }
            BenchKind::TxFixture(_) => None,
        }
    }
}

pub fn get_bench(name: &str) -> Option<Bench> {
    get_benches().into_iter().find(|b| b.name == name)
}

pub fn get_benches() -> Vec<Bench> {
    vec![
        Bench {
            name: "fibonacci",
            kind: BenchKind::Bytecode {
                bytecode: FIBONACCI.to_vec(),
                stack_input: vec![U256::from(69)],
                native: Some(|| {
                    black_box(fibonacci_rust(black_box(U256::from(70))));
                }),
                calldata: Vec::new(),
            },
        },
        // https://github.com/lambdaclass/evm_mlir/blob/b766d0bbc2093bbfa4feb3aa25baf82b512aee74/bench/revm_comparison/src/lib.rs#L12-L15
        // https://blog.lambdaclass.com/evm-performance-boosts-with-mlir/
        // > We chose 1000 as N
        Bench {
            name: "fibonacci-calldata",
            kind: BenchKind::Bytecode {
                bytecode: hex!(
                    "5f355f60015b8215601a578181019150909160019003916005565b9150505f5260205ff3"
                )
                .to_vec(),
                calldata: U256::from(1000).to_be_bytes_vec(),
                stack_input: Vec::new(),
                native: None,
            },
        },
        Bench {
            name: "factorial",
            kind: BenchKind::Bytecode {
                bytecode: hex!(
                    "5f355f60015b8215601b57906001018091029160019003916005565b9150505f5260205ff3"
                )
                .to_vec(),
                calldata: U256::from(1000).to_be_bytes_vec(),
                stack_input: Vec::new(),
                native: None,
            },
        },
        Bench {
            name: "counter",
            kind: BenchKind::Bytecode {
                bytecode: include_code_str!("../../../data/counter.rt.hex").unwrap(),
                // `increment()`
                calldata: hex!("d09de08a").to_vec(),
                stack_input: Vec::new(),
                native: None,
            },
        },
        Bench {
            name: "snailtracer",
            kind: BenchKind::Bytecode {
                bytecode: include_code_str!("../../../data/snailtracer.rt.hex").unwrap(),
                // `Benchmark()`
                calldata: hex!("30627b7c").to_vec(),
                stack_input: Vec::new(),
                native: None,
            },
        },
        Bench {
            name: "weth",
            kind: BenchKind::Bytecode {
                bytecode: include_code_str!("../../../data/weth.rt.hex").unwrap(),
                calldata: Vec::new(),
                stack_input: Vec::new(),
                native: None,
            },
        },
        Bench {
            name: "hash_10k",
            kind: BenchKind::Bytecode {
                bytecode: include_code_str!("../../../data/hash_10k.rt.hex").unwrap(),
                // `Benchmark()`
                calldata: hex!("30627b7c").to_vec(),
                stack_input: Vec::new(),
                native: None,
            },
        },
        Bench {
            name: "erc20_transfer",
            kind: BenchKind::Bytecode {
                bytecode: include_code_str!("../../../data/erc20_transfer.rt.hex").unwrap(),
                // `Benchmark()`
                calldata: hex!("30627b7c").to_vec(),
                stack_input: Vec::new(),
                native: None,
            },
        },
        Bench {
            name: "push0_proxy",
            kind: BenchKind::Bytecode {
                bytecode: include_code_str!("../../../data/push0_proxy.rt.hex").unwrap(),
                calldata: Vec::new(),
                stack_input: Vec::new(),
                native: None,
            },
        },
        Bench {
            name: "usdc_proxy",
            kind: BenchKind::Bytecode {
                bytecode: include_code_str!("../../../data/usdc_proxy.rt.hex").unwrap(),
                calldata: Vec::new(),
                stack_input: Vec::new(),
                native: None,
            },
        },
        Bench {
            name: "fiat_token",
            kind: BenchKind::Bytecode {
                bytecode: include_code_str!("../../../data/fiat_token.rt.hex").unwrap(),
                calldata: Vec::new(),
                stack_input: Vec::new(),
                native: None,
            },
        },
        Bench {
            name: "uniswap_v2_pair",
            kind: BenchKind::Bytecode {
                bytecode: include_code_str!("../../../data/uniswap_v2_pair.rt.hex").unwrap(),
                calldata: Vec::new(),
                stack_input: Vec::new(),
                native: None,
            },
        },
        Bench {
            name: "seaport",
            kind: BenchKind::Bytecode {
                bytecode: include_code_str!("../../../data/seaport.rt.hex").unwrap(),
                calldata: Vec::new(),
                stack_input: Vec::new(),
                native: None,
            },
        },
        Bench {
            name: "airdrop",
            kind: BenchKind::Bytecode {
                bytecode: include_code_str!("../../../data/airdrop.rt.hex").unwrap(),
                // `paused()`
                calldata: hex!("5c975abb").to_vec(),
                stack_input: Vec::new(),
                native: None,
            },
        },
        Bench {
            name: "bswap64",
            kind: BenchKind::Bytecode {
                bytecode: include_code_str!("../../../data/bswap64.rt.hex").unwrap(),
                // `to_little_endian_64(uint64 = 0x0102)` returns (bytes)
                calldata: hex!(
                    "ff2f79f10000000000000000000000000000000000000000000000000000000000000102"
                )
                .to_vec(),
                stack_input: Vec::new(),
                native: None,
            },
        },
        Bench {
            name: "bswap64_opt",
            kind: BenchKind::Bytecode {
                bytecode: include_code_str!("../../../data/bswap64_opt.rt.hex").unwrap(),
                // `to_little_endian_64(uint64 = 0x0102)` returns (bytes)
                calldata: hex!(
                    "ff2f79f10000000000000000000000000000000000000000000000000000000000000102"
                )
                .to_vec(),
                stack_input: Vec::new(),
                native: None,
            },
        },
        Bench {
            name: "curve_stableswap",
            kind: BenchKind::TxFixture(FixtureBenchDef {
                spec_id: revmc::primitives::hardfork::SpecId::CANCUN,
                fixture_json: include_str!("../../../data/curve-stableswap-2pool.json"),
            }),
        },
    ]
}

#[rustfmt::skip]
const FIBONACCI: &[u8] = &[
    // input to the program (which fib number we want)
    // op::PUSH2, input[0], input[1],
    op::JUMPDEST, op::JUMPDEST, op::JUMPDEST,

    // 1st/2nd fib number
    op::PUSH1, 0,
    op::PUSH1, 1,
    // 7

    // MAINLOOP:
    op::JUMPDEST,
    op::DUP3,
    op::ISZERO,
    op::PUSH1, 28, // cleanup
    op::JUMPI,

    // fib step
    op::DUP2,
    op::DUP2,
    op::ADD,
    op::SWAP2,
    op::POP,
    op::SWAP1,
    // 19

    // decrement fib step counter
    op::SWAP2,
    op::PUSH1, 1,
    op::SWAP1,
    op::SUB,
    op::SWAP2,
    op::PUSH1, 7, // goto MAINLOOP
    op::JUMP,
    // 28

    // CLEANUP:
    op::JUMPDEST,
    op::SWAP2,
    op::POP,
    op::POP,
    // done: requested fib number is the only element on the stack!
    op::STOP,
];

/// Literal translation of the `FIBONACCI` EVM bytecode to Rust.
pub fn fibonacci_rust(mut i: U256) -> U256 {
    let mut a = U256::from(0);
    let mut b = U256::from(1);
    while i != U256::ZERO {
        let tmp = a;
        a = b;
        b = b.wrapping_add(tmp);
        i -= U256::from(1);
    }
    a
}

#[test]
fn test_fibonacci_rust() {
    revm_primitives::uint! {
        assert_eq!(fibonacci_rust(0_U256), 0_U256);
        assert_eq!(fibonacci_rust(1_U256), 1_U256);
        assert_eq!(fibonacci_rust(2_U256), 1_U256);
        assert_eq!(fibonacci_rust(3_U256), 2_U256);
        assert_eq!(fibonacci_rust(4_U256), 3_U256);
        assert_eq!(fibonacci_rust(5_U256), 5_U256);
        assert_eq!(fibonacci_rust(6_U256), 8_U256);
        assert_eq!(fibonacci_rust(7_U256), 13_U256);
        assert_eq!(fibonacci_rust(8_U256), 21_U256);
        assert_eq!(fibonacci_rust(9_U256), 34_U256);
        assert_eq!(fibonacci_rust(10_U256), 55_U256);
        assert_eq!(fibonacci_rust(100_U256), 354224848179261915075_U256);
        assert_eq!(fibonacci_rust(1000_U256), 0x2e3510283c1d60b00930b7e8803c312b4c8e6d5286805fc70b594dc75cc0604b_U256);
    }
}
