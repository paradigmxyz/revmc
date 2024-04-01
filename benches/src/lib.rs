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
                black_box(fibonacci_rust(70));
            }),
            ..Default::default()
        },
        Bench {
            name: "counter",
            bytecode: hex::decode(include_str!("../../data/counter.rt.hex")).unwrap(),
            // `increment()`
            calldata: hex!("d09de08a").to_vec(),
            ..Default::default()
        },
        Bench {
            name: "snailtracer",
            bytecode: hex::decode(include_str!("../../data/snailtracer.rt.hex")).unwrap(),
            // `Benchmark()`
            calldata: hex!("30627b7c").to_vec(),
            ..Default::default()
        },
        Bench {
            name: "push0_proxy",
            bytecode: hex::decode(include_str!("../../data/push0_proxy.rt.hex")).unwrap(),
            ..Default::default()
        },
        Bench {
            name: "weth",
            bytecode: hex::decode(include_str!("../../data/weth.rt.hex")).unwrap(),
            ..Default::default()
        },
        Bench {
            name: "hash_20k",
            bytecode: hex::decode(include_str!("../../data/hash_20k.rt.hex")).unwrap(),
            // `Benchmark()`
            calldata: hex!("30627b7c").to_vec(),
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

fn fibonacci_rust(n: u16) -> U256 {
    let mut a = U256::from(0);
    let mut b = U256::from(1);
    for _ in 0..n {
        let tmp = a;
        a = b;
        b = b.wrapping_add(tmp);
    }
    a
}
