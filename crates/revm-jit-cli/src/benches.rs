use revm_jit::{interpreter::opcode as op, primitives::hex, U256};
use std::hint::black_box;

#[derive(Clone, Debug, Default)]
pub struct Bench {
    pub name: &'static str,
    pub bytecode: Vec<u8>,
    pub calldata: Vec<u8>,
    pub stack_input: Vec<U256>,
    pub native: Option<fn()>,
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
        Bench {
            name: "counter",
            bytecode: hex::decode(include_str!("../../../data/counter.rt.hex")).unwrap(),
            // `increment()`
            calldata: hex!("d09de08a").to_vec(),
            ..Default::default()
        },
        Bench {
            name: "snailtracer",
            bytecode: hex::decode(include_str!("../../../data/snailtracer.rt.hex")).unwrap(),
            // `Benchmark()`
            calldata: hex!("30627b7c").to_vec(),
            ..Default::default()
        },
        Bench {
            name: "push0_proxy",
            bytecode: hex::decode(include_str!("../../../data/push0_proxy.rt.hex")).unwrap(),
            ..Default::default()
        },
        Bench {
            name: "weth",
            bytecode: hex::decode(include_str!("../../../data/weth.rt.hex")).unwrap(),
            ..Default::default()
        },
        Bench {
            name: "hash_20k",
            bytecode: hex::decode(include_str!("../../../data/hash_20k.rt.hex")).unwrap(),
            // `Benchmark()`
            calldata: hex!("30627b7c").to_vec(),
            ..Default::default()
        },
        Bench {
            name: "fiat_token",
            bytecode: hex::decode(include_str!("../../../data/fiat_token.rt.hex")).unwrap(),
            ..Default::default()
        },
        Bench {
            name: "usdc_proxy",
            bytecode: hex::decode(include_str!("../../../data/usdc_proxy.rt.hex")).unwrap(),
            ..Default::default()
        },
        Bench {
            name: "uniswap_v2_pair",
            bytecode: hex::decode(include_str!("../../../data/uniswap_v2_pair.rt.hex")).unwrap(),
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
