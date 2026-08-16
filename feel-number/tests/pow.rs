mod common;

use dsntk_feel_number::FeelNumber;

#[test]
fn test_pow_001() {
  eqs!("1", num!(0).pow(&num!(0)).unwrap());
}

/// The one assertion in this suite where the two backends disagree, and only
/// in the 34th significant digit — the last one a decimal128 has.
///
/// A fractional exponent goes through `exp`/`ln` in both backends, and two
/// independent implementations of those do not agree on the final digit.
/// Relative difference here is 2.4e-34. The other 155 assertions in this
/// crate's suite are identical on both.
///
/// Written per backend rather than loosened to a tolerance, so that a *real*
/// divergence in `pow` still fails this test instead of hiding inside one.
#[test]
fn test_pow_002() {
  #[cfg(not(feature = "use-fastnum"))]
  eqs!("41959.857373594361860953310707468", num!(12.2384283).pow(&num!(4.25)).unwrap());
  #[cfg(feature = "use-fastnum")]
  eqs!("41959.85737359436186095331070746801", num!(12.2384283).pow(&num!(4.25)).unwrap());
}

#[test]
fn test_pow_003() {
  assert!(num!(9999).pow(&num!(9999)).is_none());
}
