use revmc::SpecId;
use std::borrow::Cow;

/// Benchmark definition.
#[derive(Clone, Debug, Default)]
pub struct Bench {
    pub name: &'static str,
    /// Transaction fixture JSON.
    pub fixture_json: Option<Cow<'static, str>>,
    /// Spec ID for transaction fixture benchmarks.
    pub spec_id: Option<SpecId>,
}

impl Bench {
    /// Whether this is a transaction-fixture benchmark.
    pub fn is_fixture(&self) -> bool {
        self.fixture_json.is_some()
    }
}

pub fn fixture_from_bytecode(name: &'static str, bytecode: Vec<u8>, spec_id: SpecId) -> Bench {
    let bytecode = revmc::primitives::hex::encode(bytecode);
    Bench {
        name,
        fixture_json: Some(Cow::Owned(format!(
            r#"{{
  "{name}": {{
    "env": {{
      "currentBaseFee": "0x0",
      "currentCoinbase": "0x0000000000000000000000000000000000000000",
      "currentGasLimit": "0x7fffffffffffffff",
      "currentNumber": "0x1",
      "currentTimestamp": "0x1"
    }},
    "pre": {{
      "0x1111111111111111111111111111111111111111": {{
        "balance": "0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "code": "0x",
        "nonce": "0x0",
        "storage": {{}}
      }},
      "0xcccccccccccccccccccccccccccccccccccccccc": {{
        "balance": "0x0",
        "code": "0x{bytecode}",
        "nonce": "0x1",
        "storage": {{}}
      }}
    }},
    "transaction": [
      {{
        "data": "0x",
        "gasLimit": "0x7fffffffffffffff",
        "gasPrice": "0x0",
        "nonce": "0x0",
        "sender": "0x1111111111111111111111111111111111111111",
        "to": "0xcccccccccccccccccccccccccccccccccccccccc",
        "value": "0x0"
      }}
    ]
  }}
}}"#
        ))),
        spec_id: Some(spec_id),
    }
}

pub fn get_bench(name: &str) -> Option<Bench> {
    get_benches().into_iter().find(|b| b.name == name)
}

macro_rules! bench {
    ($name:literal, $spec:expr, $path:literal) => {
        Bench {
            name: $name,
            spec_id: Some($spec),
            fixture_json: Some(Cow::Borrowed(include_str!($path))),
        }
    };
}

pub fn get_benches() -> Vec<Bench> {
    vec![
        bench!("fibonacci-calldata", SpecId::OSAKA, "../../../data/fibonacci-calldata.json"),
        bench!("factorial", SpecId::OSAKA, "../../../data/factorial.json"),
        bench!("counter", SpecId::OSAKA, "../../../data/counter.json"),
        bench!("snailtracer", SpecId::OSAKA, "../../../data/snailtracer.json"),
        bench!("weth", SpecId::OSAKA, "../../../data/weth.json"),
        bench!("hash_10k", SpecId::OSAKA, "../../../data/hash_10k.json"),
        bench!("erc20_transfer", SpecId::OSAKA, "../../../data/erc20_transfer.json"),
        bench!("push0_proxy", SpecId::OSAKA, "../../../data/push0_proxy.json"),
        bench!("usdc_proxy", SpecId::OSAKA, "../../../data/usdc_proxy.json"),
        bench!("fiat_token", SpecId::OSAKA, "../../../data/fiat_token.json"),
        bench!("uniswap_v2_pair", SpecId::OSAKA, "../../../data/uniswap_v2_pair.json"),
        bench!("univ2_router", SpecId::OSAKA, "../../../data/univ2_router.json"),
        bench!("seaport", SpecId::OSAKA, "../../../data/seaport.json"),
        bench!("airdrop", SpecId::OSAKA, "../../../data/airdrop.json"),
        bench!("bswap64", SpecId::OSAKA, "../../../data/bswap64.json"),
        bench!("bswap64_opt", SpecId::OSAKA, "../../../data/bswap64_opt.json"),
        bench!("eip4788", SpecId::OSAKA, "../../../data/eip4788.json"),
        bench!("eip2935", SpecId::OSAKA, "../../../data/eip2935.json"),
        bench!("curve_stableswap", SpecId::CANCUN, "../../../data/curve-stableswap-2pool.json"),
        bench!("onchain_lm_v2", SpecId::CANCUN, "../../../data/onchain-lm-v2.json"),
        bench!("burntpix", SpecId::CANCUN, "../../../data/burntpix.json"),
    ]
}

#[cfg(all(test, feature = "llvm"))]
mod tests {
    use super::get_bench;
    use crate::PreparedBench;

    #[test]
    fn selected_contract_benches_succeed() {
        for name in ["fiat_token", "uniswap_v2_pair", "seaport"] {
            let bench = get_bench(name).unwrap();
            let (prepared, _compiler) = PreparedBench::load(&bench, bench.spec_id.unwrap());
            let result = prepared.run_interpreter();
            assert!(result.result.is_success(), "{name}: {:?}", result.result);
        }
    }
}
