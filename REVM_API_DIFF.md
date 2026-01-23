# REVM API Differences: v34.0.0 → Latest Main

This document details the breaking changes between revm v34.0.0 (currently used by revmc) and the latest revm main branch.

## Summary

| Current revmc | Latest Main |
|---------------|-------------|
| revm v34.0.0 | ~v35+ (main branch) |
| revm-interpreter v32.0.0 | Latest |
| revm-context-interface v14.0.0 | context-interface (renamed package) |
| revm-primitives v22.0.0 | primitives |

## Package Reorganization

The latest revm has reorganized its crates:
- `revm-context-interface` → `context-interface` (crate path: `crates/context/interface/`)
- `revm-primitives` → `primitives`
- Gas constants/functions moved from `revm-interpreter` to `context-interface::cfg::gas`

---

## 1. Host Trait Changes

### Location
- **v34**: `revm_context_interface::host::Host`
- **Latest**: `context_interface::host::Host`

### New Method: `slot_num()`

```rust
// NEW in latest main
fn slot_num(&self) -> U256;
```

**Migration**: Implement this new required method in any custom `Host` implementations.

### `selfdestruct()` Signature (Already in v34)

```rust
// Current (v34.0.0) - already updated
fn selfdestruct(
    &mut self,
    address: Address,
    target: Address,
    skip_cold_load: bool,  // Added for BAL support
) -> Result<StateLoad<SelfDestructResult>, LoadError>;

// Same in latest main
```

**Note**: revmc already adapted to this in v34. No changes needed.

---

## 2. CreateInputs Changes

### Location
- `revm_interpreter::CreateInputs`

### Fields Made Private with Accessors

```rust
// OLD (pre-v34 style - direct field access)
pub struct CreateInputs {
    pub caller: Address,
    pub scheme: CreateScheme,
    pub value: U256,
    pub init_code: Bytes,
    pub gas_limit: u64,
}

// NEW (v34+)
pub struct CreateInputs {
    caller: Address,           // private
    scheme: CreateScheme,      // private
    value: U256,               // private
    init_code: Bytes,          // private
    gas_limit: u64,            // private
    cached_address: OnceCell<Address>,  // NEW: cached computed address
}

// Use constructor and getters:
impl CreateInputs {
    pub fn new(
        caller: Address,
        scheme: CreateScheme,
        value: U256,
        init_code: Bytes,
        gas_limit: u64,
    ) -> Self;

    pub fn created_address(&self, nonce: u64) -> Address;  // Cached!
    pub fn caller(&self) -> Address;
    pub fn scheme(&self) -> CreateScheme;
    pub fn value(&self) -> U256;
    pub fn init_code(&self) -> &Bytes;
    pub fn gas_limit(&self) -> u64;

    // Setters
    pub fn set_call(&mut self, caller: Address);
    pub fn set_scheme(&mut self, scheme: CreateScheme);
    pub fn set_value(&mut self, value: U256);
    pub fn set_init_code(&mut self, init_code: Bytes);
    pub fn set_gas_limit(&mut self, gas_limit: u64);
}
```

### Migration in revmc-builtins/src/lib.rs

```rust
// OLD (if using field initialization)
CreateInputs {
    caller: ecx.input.target_address,
    scheme,
    value,
    init_code: code,
    gas_limit,
}

// NEW (already correct in current revmc code)
CreateInputs::new(ecx.input.target_address, scheme, value, code, gas_limit)
```

**Status**: ✅ revmc already uses `CreateInputs::new()` constructor.

---

## 3. CallInputs Changes

### `known_bytecode` Field

```rust
// v34.0.0+
pub struct CallInputs {
    pub input: CallInput,
    pub return_memory_offset: Range<usize>,
    pub gas_limit: u64,
    pub bytecode_address: Address,
    pub known_bytecode: Option<(B256, Bytecode)>,  // NEW in v34
    pub target_address: Address,
    pub caller: Address,
    pub value: CallValue,
    pub scheme: CallScheme,
    pub is_static: bool,
}
```

**Status**: ✅ revmc already sets `known_bytecode: None`.

---

## 4. GasParams - Dynamic Gas Configuration

### Location Change
- **v34**: `revm_context_interface::cfg::GasParams`
- **Latest**: `context_interface::cfg::GasParams` / `context_interface::cfg::gas_params`

### New Dynamic Gas System

```rust
// Access via Host trait
fn gas_params(&self) -> &GasParams;

// GasParams provides dynamic gas constants
impl GasParams {
    pub fn new_spec(spec: SpecId) -> Self;
    pub fn get(&self, id: GasId) -> u64;
    pub fn override_gas(&mut self, values: impl IntoIterator<Item = (GasId, u64)>);

    // Convenience methods for specific calculations
    pub fn memory_gas(&self, new_num: usize) -> u64;
    pub fn sstore_cost(&self, result: &SStoreResult, remaining_gas: u64, is_cold: bool) -> Option<u64>;
    pub fn sstore_refund(&self, result: &SStoreResult) -> i64;
    pub fn selfdestruct_cost(&self, should_charge_topup: bool, is_cold: bool) -> u64;
    pub fn selfdestruct_refund(&self) -> i64;
    // ... many more
}

// GasId enum for dynamic gas lookup
pub struct GasId(u8);
impl GasId {
    pub const fn exp_byte_gas() -> GasId;
    pub const fn memory_linear_cost() -> GasId;
    pub const fn memory_quadratic_reduction() -> GasId;
    pub const fn cold_account_additional_cost() -> GasId;
    // ... 38 different gas IDs
}
```

### revmc Usage Updates

Current revmc code in `revmc-builtins/src/lib.rs`:
```rust
// Current pattern (line 857-860)
gas!(ecx, ecx.host.gas_params().selfdestruct_cost(should_charge_topup, res.is_cold));
ecx.gas.record_refund(ecx.host.gas_params().selfdestruct_refund());
```

**Status**: ✅ Already using the new GasParams API correctly.

---

## 5. Gas Constants Location

### Import Path Changes

```rust
// OLD - from revm-interpreter
use revm_interpreter::gas::{
    COLD_ACCOUNT_ACCESS_COST,
    COLD_SLOAD_COST,
    WARM_STORAGE_READ_COST,
    CALL_STIPEND,
    // etc.
};

// NEW - re-exported through revm-interpreter
// (actually from context_interface::cfg::gas)
use revm_interpreter::gas::*;  // Still works, re-exports from context_interface
```

### Constants Available

Both v34 and latest provide these constants via `revm_interpreter::gas`:
- `ZERO`, `BASE`, `VERYLOW`, `LOW`, `MID`, `HIGH`
- `CREATE`, `CALLVALUE`, `NEWACCOUNT`
- `COLD_SLOAD_COST`, `COLD_ACCOUNT_ACCESS_COST`
- `WARM_STORAGE_READ_COST`, `WARM_SSTORE_RESET`
- `SSTORE_SET`, `SSTORE_RESET`, `REFUND_SSTORE_CLEARS`
- `CALL_STIPEND`, `SELFDESTRUCT_REFUND`
- `KECCAK256`, `KECCAK256WORD`, `COPY`
- `LOG`, `LOGDATA`, `LOGTOPIC`
- `INITCODE_WORD_COST`, `CODEDEPOSIT`
- `ISTANBUL_SLOAD_GAS`

**Status**: ✅ revmc-builtins/src/gas.rs already uses `pub use revm_interpreter::gas::*;`

---

## 6. Removed/Changed Macros

### `tri!`, `gas_or_fail!`, `otry!` Removed

These macros were removed from `revm-interpreter`.

**revmc Status**: revmc defines its own `tri!` macro in `revmc-builtins/src/gas.rs`:
```rust
macro_rules! tri {
    ($e:expr) => {
        match $e {
            Some(v) => v,
            None => return None,
        }
    };
}
```
✅ No migration needed.

---

## 7. MemoryGas API Changes

### `record_new_len()` Signature

```rust
// v34.0.0+
impl MemoryGas {
    pub fn record_new_len(
        &mut self,
        new_num: usize,
        linear_cost: u64,      // NEW: configurable
        quadratic_cost: u64,   // NEW: configurable
    ) -> Option<u64>;
}

// Usage pattern
let linear = gas_params.get(GasId::memory_linear_cost());
let quadratic = gas_params.get(GasId::memory_quadratic_reduction());
memory_gas.record_new_len(new_words, linear, quadratic);
```

### revmc Usage

Check `revmc-builtins/src/utils.rs` for memory resize logic.

---

## 8. LoadError Enum

```rust
// v34.0.0+
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum LoadError {
    DBError,
    ColdLoadSkipped,  // For BAL (EIP-7928) support
}
```

Used in:
- `Host::selfdestruct()` → `Result<StateLoad<SelfDestructResult>, LoadError>`
- `Host::sstore_skip_cold_load()` → `Result<StateLoad<SStoreResult>, LoadError>`
- `Host::sload_skip_cold_load()` → `Result<StateLoad<StorageValue>, LoadError>`
- `Host::load_account_info_skip_cold_load()` → `Result<AccountInfoLoad<'_>, LoadError>`

---

## 9. New Imports Required for Latest Main

```rust
// If upgrading to latest main, update import paths:

// Primitives
use primitives::{Address, Bytes, U256, B256, Log};  // instead of revm_primitives

// Context interface
use context_interface::host::Host;
use context_interface::cfg::{GasParams, GasId};
use context_interface::cfg::gas::*;  // gas constants

// Interpreter (unchanged)
use interpreter::{
    CallInput, CallInputs, CallValue, CallScheme,
    CreateInputs, CreateScheme,
    InstructionResult, InterpreterAction, InterpreterResult,
    Gas, MemoryGas,
};
```

---

## 10. Action Items for revmc

### Already Correct ✅
1. `CreateInputs::new()` constructor usage
2. `CallInputs` with `known_bytecode: None`
3. `Host::selfdestruct()` with `skip_cold_load` parameter
4. `GasParams` usage via `ecx.host.gas_params()`
5. Gas constants from `revm_interpreter::gas::*`
6. Custom `tri!` macro definition

### May Need Updates ⚠️
1. If implementing custom `Host`: add `fn slot_num(&self) -> U256`
2. Memory gas calculations may need `GasParams` dynamic values
3. Any direct field access to `CreateInputs` (use getters instead)

### Version Pinning
Current Cargo.toml is correctly pinned:
```toml
revm = { version = "34.0", default-features = false }
revm-primitives = { version = "22.0", default-features = false }
revm-interpreter = { version = "32.0", default-features = false }
revm-context-interface = { version = "14.0", default-features = false }
```

---

## Future Migration Notes (v34 → v35+)

When upgrading beyond v34, expect:
1. Package rename: `revm-context-interface` → `context-interface`
2. Package rename: `revm-primitives` → `primitives`
3. Package rename: `revm-interpreter` → `interpreter`
4. New `Host::slot_num()` method required
5. Possible further GasParams API expansion for new EIPs
