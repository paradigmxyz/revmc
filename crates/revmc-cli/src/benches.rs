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

/// Benchmark definition. Bytecode benchmarks use the `bytecode` / `calldata` /
/// `stack_input` fields; transaction-fixture benchmarks use `fixture_json` /
/// `spec_id` instead.
#[derive(Clone, Debug, Default)]
pub struct Bench {
    pub name: &'static str,
    pub bytecode: Vec<u8>,
    pub calldata: Vec<u8>,
    pub stack_input: Vec<U256>,
    pub native: Option<fn()>,
    /// Pre-seeded storage entries `(key, value)` inserted at `Address::ZERO`.
    pub storage: Vec<(U256, U256)>,
    /// Override for `Host::block_number()`.
    pub block_number: Option<U256>,
    /// Override for `Host::timestamp()`.
    pub timestamp: Option<U256>,
    /// Transaction fixture JSON (full-EVM benchmark).
    pub fixture_json: Option<&'static str>,
    /// Spec ID for transaction fixture benchmarks.
    pub spec_id: Option<revmc::primitives::hardfork::SpecId>,
}

impl Bench {
    /// Whether this is a transaction-fixture benchmark.
    pub fn is_fixture(&self) -> bool {
        self.fixture_json.is_some()
    }
}

pub fn get_bench(name: &str) -> Option<Bench> {
    get_benches().into_iter().find(|b| b.name == name)
}

pub fn get_benches() -> Vec<Bench> {
    vec![
        Bench {
            name: "fibonacci",
            bytecode: FIBONACCI.to_vec(),
            stack_input: vec![U256::from(69)],
            native: Some(|| {
                black_box(fibonacci_rust(black_box(U256::from(70))));
            }),
            ..Default::default()
        },
        // https://github.com/lambdaclass/evm_mlir/blob/b766d0bbc2093bbfa4feb3aa25baf82b512aee74/bench/revm_comparison/src/lib.rs#L12-L15
        // https://blog.lambdaclass.com/evm-performance-boosts-with-mlir/
        // > We chose 1000 as N
        Bench {
            name: "fibonacci-calldata",
            bytecode: hex!(
                "5f355f60015b8215601a578181019150909160019003916005565b9150505f5260205ff3"
            )
            .to_vec(),
            calldata: U256::from(1000).to_be_bytes_vec(),
            ..Default::default()
        },
        Bench {
            name: "factorial",
            bytecode: hex!(
                "5f355f60015b8215601b57906001018091029160019003916005565b9150505f5260205ff3"
            )
            .to_vec(),
            calldata: U256::from(1000).to_be_bytes_vec(),
            ..Default::default()
        },
        Bench {
            name: "counter",
            bytecode: include_code_str!("../../../data/counter.rt.hex").unwrap(),
            // `increment()`
            calldata: hex!("d09de08a").to_vec(),
            ..Default::default()
        },
        Bench {
            name: "snailtracer",
            bytecode: include_code_str!("../../../data/snailtracer.rt.hex").unwrap(),
            // `Benchmark()`
            calldata: hex!("30627b7c").to_vec(),
            ..Default::default()
        },
        Bench {
            name: "weth",
            bytecode: include_code_str!("../../../data/weth.rt.hex").unwrap(),
            ..Default::default()
        },
        Bench {
            name: "hash_10k",
            bytecode: include_code_str!("../../../data/hash_10k.rt.hex").unwrap(),
            // `Benchmark()`
            calldata: hex!("30627b7c").to_vec(),
            ..Default::default()
        },
        Bench {
            name: "erc20_transfer",
            bytecode: include_code_str!("../../../data/erc20_transfer.rt.hex").unwrap(),
            // `Benchmark()`
            calldata: hex!("30627b7c").to_vec(),
            ..Default::default()
        },
        Bench {
            name: "push0_proxy",
            bytecode: include_code_str!("../../../data/push0_proxy.rt.hex").unwrap(),
            ..Default::default()
        },
        Bench {
            name: "usdc_proxy",
            bytecode: include_code_str!("../../../data/usdc_proxy.rt.hex").unwrap(),
            ..Default::default()
        },
        Bench {
            name: "fiat_token",
            bytecode: include_code_str!("../../../data/fiat_token.rt.hex").unwrap(),
            ..Default::default()
        },
        Bench {
            name: "uniswap_v2_pair",
            bytecode: include_code_str!("../../../data/uniswap_v2_pair.rt.hex").unwrap(),
            ..Default::default()
        },
        Bench {
            name: "univ2_router",
            bytecode: include_code_str!("../../../data/univ2_router.rt.hex").unwrap(),
            ..Default::default()
        },
        Bench {
            name: "seaport",
            bytecode: include_code_str!("../../../data/seaport.rt.hex").unwrap(),
            ..Default::default()
        },
        Bench {
            name: "airdrop",
            bytecode: include_code_str!("../../../data/airdrop.rt.hex").unwrap(),
            // `paused()`
            calldata: hex!("5c975abb").to_vec(),
            ..Default::default()
        },
        Bench {
            name: "bswap64",
            bytecode: include_code_str!("../../../data/bswap64.rt.hex").unwrap(),
            // `to_little_endian_64(uint64 = 0x0102)` returns (bytes)
            calldata: hex!(
                "ff2f79f10000000000000000000000000000000000000000000000000000000000000102"
            )
            .to_vec(),
            ..Default::default()
        },
        Bench {
            name: "bswap64_opt",
            bytecode: include_code_str!("../../../data/bswap64_opt.rt.hex").unwrap(),
            // `to_little_endian_64(uint64 = 0x0102)` returns (bytes)
            calldata: hex!(
                "ff2f79f10000000000000000000000000000000000000000000000000000000000000102"
            )
            .to_vec(),
            ..Default::default()
        },
        // EIP-4788 beacon block root contract.
        // https://eips.ethereum.org/EIPS/eip-4788
        Bench {
            name: "eip4788",
            bytecode: hex!(
                "3373fffffffffffffffffffffffffffffffffffffffe14604d57602036146024575f5ffd5b5f35801560495762001fff810690815414603c575f5ffd5b62001fff01545f5260205ff35b5f5ffd5b62001fff42064281555f359062001fff015500"
            )
            .to_vec(),
            // `get` path: 32-byte timestamp query.
            calldata: U256::from(1774396307).to_be_bytes_vec(),
            // ring_index = 1774396307 % 8191 = 4550
            // slot 4550: stored timestamp must equal query.
            // slot 4550 + 8191 = 12741: the beacon block root to return.
            storage: vec![
                (U256::from(4550), U256::from(1774396307)),
                (U256::from(12741), U256::from(0xbeacu64)),
            ],
            ..Default::default()
        },
        // EIP-2935 historical block hashes contract.
        // https://eips.ethereum.org/EIPS/eip-2935
        Bench {
            name: "eip2935",
            bytecode: hex!(
                "3373fffffffffffffffffffffffffffffffffffffffe14604657602036036042575f35600143038111604257611fff81430311604257611fff9006545f5260205ff35b5f5ffd5b5f35611fff60014303065500"
            )
            .to_vec(),
            // `get` path: 32-byte block number query.
            calldata: U256::from(1).to_be_bytes_vec(),
            // Needs NUMBER > query (1) and NUMBER - query <= 8191.
            // ring_index = 1 % 8191 = 1; slot 1 holds the block hash.
            storage: vec![(U256::from(1), U256::from(0xb10c_ba5eu64))],
            block_number: Some(U256::from(100)),
            ..Default::default()
        },
        Bench {
            name: "curve_stableswap",
            spec_id: Some(revmc::primitives::hardfork::SpecId::CANCUN),
            fixture_json: Some(include_str!("../../../data/curve-stableswap-2pool.json")),
            ..Default::default()
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
