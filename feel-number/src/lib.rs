#![doc = include_str!("../docs/README.md")]
#![deny(rustdoc::broken_intra_doc_links)]

#[macro_use]
extern crate dsntk_macros;

mod errors;

// One of two decimal128 backends, chosen by feature (see `Cargo.toml`). The
// two are API-identical: `FeelNumber` is this crate's entire public surface,
// so nothing above it needs to know which one it got.
//
// `use-fastnum` wins when both are enabled, and that tie-break is the reason
// these are two additive features rather than one exclusive choice.

#[cfg(feature = "use-fastnum")]
mod fastnum_number;
#[cfg(feature = "use-fastnum")]
pub use fastnum_number::FeelNumber;

#[cfg(all(feature = "use-dfp", not(feature = "use-fastnum")))]
mod dfp_number;
#[cfg(all(feature = "use-dfp", not(feature = "use-fastnum")))]
pub use dfp_number::FeelNumber;

#[cfg(not(any(feature = "use-fastnum", feature = "use-dfp")))]
compile_error!(
  "dsntk-feel-number needs a decimal backend: enable either `use-fastnum` (pure Rust, reaches wasm32) \
   or `use-dfp` (Intel's decimal C library, needs a C toolchain). `use-fastnum` is the default, so \
   this error means default-features was switched off without naming a replacement."
);
