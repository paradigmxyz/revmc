//! EOF stub types.
//!
//! EOF (EVM Object Format) has been removed from revm v34.
//! These are stub types to maintain API compatibility.
//! EOF functionality is disabled at runtime.

use revm_primitives::Bytes;

// EOF opcode constants (removed from revm-bytecode in v34)
// These are kept for API compatibility but EOF is not supported at runtime.

/// DATALOAD opcode (0xD0)
pub const DATALOAD: u8 = 0xD0;
/// DATALOADN opcode (0xD1)
pub const DATALOADN: u8 = 0xD1;
/// DATASIZE opcode (0xD2)
pub const DATASIZE: u8 = 0xD2;
/// DATACOPY opcode (0xD3)
pub const DATACOPY: u8 = 0xD3;
/// RJUMP opcode (0xE0)
pub const RJUMP: u8 = 0xE0;
/// RJUMPI opcode (0xE1)
pub const RJUMPI: u8 = 0xE1;
/// RJUMPV opcode (0xE2)
pub const RJUMPV: u8 = 0xE2;
/// CALLF opcode (0xE3)
pub const CALLF: u8 = 0xE3;
/// RETF opcode (0xE4)
pub const RETF: u8 = 0xE4;
/// JUMPF opcode (0xE5)
pub const JUMPF: u8 = 0xE5;
/// DUPN opcode (0xE6)
pub const DUPN: u8 = 0xE6;
/// SWAPN opcode (0xE7)
pub const SWAPN: u8 = 0xE7;
/// EXCHANGE opcode (0xE8)
pub const EXCHANGE: u8 = 0xE8;
/// EOFCREATE opcode (0xEC)
pub const EOFCREATE: u8 = 0xEC;
/// RETURNCONTRACT opcode (0xEE)
pub const RETURNCONTRACT: u8 = 0xEE;
/// RETURNDATALOAD opcode (0xF7)
pub const RETURNDATALOAD: u8 = 0xF7;
/// EXTCALL opcode (0xF8)
pub const EXTCALL: u8 = 0xF8;
/// EXTDELEGATECALL opcode (0xF9)
pub const EXTDELEGATECALL: u8 = 0xF9;
/// EXTSTATICCALL opcode (0xFB)
pub const EXTSTATICCALL: u8 = 0xFB;

/// EOF magic bytes (0xEF00).
pub const EOF_MAGIC_BYTES: [u8; 2] = [0xEF, 0x00];

/// Stub EOF container.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Eof {
    /// EOF header.
    pub header: EofHeader,
    /// EOF body.
    pub body: EofBody,
    /// Raw bytes of the EOF container.
    pub raw: Bytes,
}

impl Eof {
    /// Decode EOF from bytes. Always returns an error since EOF is not supported.
    pub fn decode(_bytes: Bytes) -> Result<Self, EofDecodeError> {
        Err(EofDecodeError::NotSupported)
    }

    /// Returns a slice of the data section.
    pub fn data_slice(&self, offset: usize, len: usize) -> &[u8] {
        let data = &self.body.data_section;
        if offset >= data.len() {
            return &[];
        }
        let end = (offset + len).min(data.len());
        &data[offset..end]
    }
}

/// Stub EOF header.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EofHeader {
    /// Sum of all code section sizes.
    pub sum_code_sizes: usize,
    /// Types section size.
    pub types_size: u16,
    /// Code section sizes.
    pub code_sizes: Vec<u16>,
    /// Container section sizes.
    pub container_sizes: Vec<u16>,
    /// Data section size.
    pub data_size: u16,
}

impl EofHeader {
    /// Returns the number of code sections.
    pub fn code_sections_len(&self) -> usize {
        self.code_sizes.len()
    }
}

/// Stub EOF body.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EofBody {
    /// Types section.
    pub types_section: Vec<TypesSection>,
    /// Code sections.
    pub code_section: Vec<Bytes>,
    /// Container sections.
    pub container_section: Vec<Bytes>,
    /// Data section.
    pub data_section: Bytes,
}

impl EofBody {
    /// Converts this body into a full EOF container.
    /// Since EOF is not fully supported, this creates a minimal stub container.
    pub fn into_eof(self) -> Eof {
        // Build the raw bytes for the EOF container
        let mut raw = Vec::new();
        raw.extend_from_slice(&EOF_MAGIC_BYTES);
        raw.push(0x01); // version
        
        // Simplified header - just encode the code sections
        let code_sizes: Vec<u16> = self.code_section.iter().map(|c| c.len() as u16).collect();
        let types_size = (self.types_section.len() * 4) as u16;
        let sum_code_sizes: usize = code_sizes.iter().map(|&s| s as usize).sum();
        
        // Types section header (kind=1)
        raw.push(0x01);
        raw.extend_from_slice(&types_size.to_be_bytes());
        
        // Code section header (kind=2)
        raw.push(0x02);
        raw.extend_from_slice(&(code_sizes.len() as u16).to_be_bytes());
        for &size in &code_sizes {
            raw.extend_from_slice(&size.to_be_bytes());
        }
        
        // Container section header (kind=3) if any
        if !self.container_section.is_empty() {
            raw.push(0x03);
            raw.extend_from_slice(&(self.container_section.len() as u16).to_be_bytes());
            for c in &self.container_section {
                raw.extend_from_slice(&(c.len() as u16).to_be_bytes());
            }
        }
        
        // Data section header (kind=4)
        raw.push(0x04);
        raw.extend_from_slice(&(self.data_section.len() as u16).to_be_bytes());
        
        // Terminator
        raw.push(0x00);
        
        // Types section body
        for ts in &self.types_section {
            raw.push(ts.inputs);
            raw.push(ts.outputs);
            raw.extend_from_slice(&ts.max_stack_size.to_be_bytes());
        }
        
        // Code sections body
        for code in &self.code_section {
            raw.extend_from_slice(code);
        }
        
        // Container sections body
        for container in &self.container_section {
            raw.extend_from_slice(container);
        }
        
        // Data section body
        raw.extend_from_slice(&self.data_section);
        
        Eof {
            header: EofHeader {
                sum_code_sizes,
                types_size,
                code_sizes,
                container_sizes: self.container_section.iter().map(|c| c.len() as u16).collect(),
                data_size: self.data_section.len() as u16,
            },
            body: self,
            raw: Bytes::from(raw),
        }
    }
}

/// Stub types section.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TypesSection {
    /// Number of inputs.
    pub inputs: u8,
    /// Number of outputs.
    pub outputs: u8,
    /// Maximum stack height.
    pub max_stack_size: u16,
}

/// EOF decode error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EofDecodeError {
    /// EOF is not supported in this version.
    NotSupported,
}

impl std::fmt::Display for EofDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EofDecodeError::NotSupported => write!(f, "EOF is not supported in revm v34"),
        }
    }
}

impl std::error::Error for EofDecodeError {}


