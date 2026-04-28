use revmc::{U256, primitives::hex};

macro_rules! include_code_str {
    ($path:literal) => {
        crate::read_code_string(
            include_bytes!($path),
            std::path::Path::new($path).extension().and_then(|s| s.to_str()),
        )
    };
}

/// Benchmark definition. Bytecode benchmarks use the `bytecode` / `calldata`
/// fields; transaction-fixture benchmarks use `fixture_json` / `spec_id`
/// instead.
#[derive(Clone, Debug, Default)]
pub struct Bench {
    pub name: &'static str,
    pub bytecode: Vec<u8>,
    pub calldata: Vec<u8>,
    /// Pre-seeded storage entries `(key, value)` inserted at `BENCH_CONTRACT`.
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
            // `transfer(address(1), 1000)`
            calldata: hex!(
                "a9059cbb"
                "0000000000000000000000000000000000000000000000000000000000000001"
                "00000000000000000000000000000000000000000000000000000000000003e8"
            )
            .to_vec(),
            // balanceOf[BENCH_CALLER] = 1_000_000
            // slot = keccak256(abi.encode(0x1111..1111, uint256(3)))
            storage: vec![(
                U256::from_str_radix(
                    "fc40ea33816453f766ebc0872d4b5152b468882abe7b6b35528069db4d6e41c4",
                    16,
                )
                .unwrap(),
                U256::from(1_000_000),
            )],
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
            // `getAmountOut(1_000_000, 100_000_000_000, 50_000_000_000)`
            calldata: hex!(
                "054d50d4"
                "00000000000000000000000000000000000000000000000000000000000f4240"
                "000000000000000000000000000000000000000000000000000000174876e800"
                "0000000000000000000000000000000000000000000000000000000ba43b7400"
            )
            .to_vec(),
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
        // https://etherscan.io/address/0x5F8E7D750E75b44747C058A204D8DEa0D18fA5d3
        Bench {
            name: "onchain_lm_v2",
            spec_id: Some(revmc::primitives::hardfork::SpecId::CANCUN),
            fixture_json: Some(include_str!("../../../data/onchain-lm-v2.json")),
            ..Default::default()
        },
        // https://github.com/karalabe/burntpix-benchmark
        Bench {
            name: "burntpix",
            spec_id: Some(revmc::primitives::hardfork::SpecId::CANCUN),
            fixture_json: Some(include_str!("../../../data/burntpix.json")),
            ..Default::default()
        },
    ]
}
