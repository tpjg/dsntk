//! The FEEL number type, over `fastnum` instead of Intel's C library.
//!
//! Every item mirrors `dsntk-feel-number` 0.3.0. Where a method looks odd,
//! e.g. `even` not checking integrality while `odd` does, the comments say
//! which upstream call each line stands in for.

use crate::errors::*;
use dsntk_common::{DsntkError, Jsonify, Result};
use fastnum::decimal::{Context, Decimal, RoundingMode};
use std::cmp::Ordering;
use std::fmt::{self, Debug, Display};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};
use std::str::FromStr;

/// The working type: fastnum's 256-bit decimal (~77 significant digits).
///
/// Deliberately wider than decimal128. fastnum has no precision *setting* —
/// its capacity is the type width — so 34-digit results are produced by
/// rounding in [`norm`] after each operation. Computing intermediates at 77
/// digits means the exact result of `a op b` fits for every ordinary operand
/// pair, so that rounding happens **once**; a 128-bit type would round to ~38
/// digits first and to 34 second, and double rounding is a divergence you
/// cannot test for afterwards.
type D = Decimal<4>;

/// decimal128's significand: 34 decimal digits.
const PRECISION: i32 = 34;

/// decimal128's largest adjusted exponent — above it, values are infinity.
const MAX_ADJUSTED_EXPONENT: i32 = 6144;

/// decimal128's smallest adjusted exponent, below which a value underflows to
/// zero. (Between this and -6143 the format degrades gradually into
/// subnormals; that is the part not emulated.)
const MIN_ADJUSTED_EXPONENT: i32 = -6176;

/// The largest exponent decimal128 can *store*. Above it the same value is
/// held in a different cohort — a wider coefficient against a smaller
/// exponent — which is why [`norm`] pads rather than leaves it alone.
const MAX_STORED_EXPONENT: i16 = 6111;

/// Scale bounds `validate_scale` enforces, from upstream verbatim. Note they
/// are not symmetric and do not match the range named in upstream's own doc
/// comment; they are what the constants say, and the vendored tests pin them.
const MIN_SCALE: i32 = -6111;
const MAX_SCALE: i32 = 6144;

/// Half-even, and **no traps**: fastnum panics on signalled operations by
/// default, whereas the C library returns infinity or NaN and lets the caller
/// look. Every value we construct carries this context so the behaviour
/// propagates through arithmetic.
const CTX: Context = Context::default().with_rounding_mode(RoundingMode::HalfEven).without_traps();

/// The adjusted exponent: the power of ten of the leading digit. Stands in for
/// `bid128_ilogb`, **including its answer for zero**, which is `i32::MIN` —
/// that is what makes `ceiling(0, 1)` an error upstream rather than `0`, and
/// callers depend on the refusal.
fn ilogb(d: D) -> i32 {
  if d.is_zero() || !d.is_finite() {
    return i32::MIN;
  }
  d.digits_count() as i32 - 1 - d.fractional_digits_count() as i32
}

/// log10 of the magnitude, as an f64 — used only to decide whether a result
/// is inside decimal128's range at all.
///
/// Deliberately not `f64::try_from(d).log10()`: that saturates to infinity
/// above 1e308, so `1e6000 ** 0.5` — an ordinary 1e3000 — would be judged an
/// overflow and refused. The exponent is exact and only the mantissa goes
/// through f64.
fn log10_abs(d: D) -> f64 {
  if d.is_zero() || !d.is_finite() {
    return f64::NEG_INFINITY;
  }
  let digits = d.digits().to_string();
  let mantissa: f64 = if digits.len() > 1 {
    format!("{}.{}", &digits[..1], &digits[1..digits.len().min(17)])
  } else {
    digits.clone()
  }
  .parse()
  .unwrap_or(1.0);
  ilogb(d) as f64 + mantissa.log10()
}

/// Bring a raw fastnum result back to decimal128: 34 significant digits,
/// half-even, overflowing to infinity.
///
/// Underflow is *not* emulated — decimal128 loses precision gradually below
/// 1e-6143 and fastnum does not. See the deviation table in `docs/dmn.md`.
fn norm(d: D) -> D {
  let d = d.with_ctx(CTX);
  // NaN has no sign in the C library's rendering — `bid128_to_string`
  // answers `+NaN` however the value arose — while fastnum carries the
  // sign of the operand through, so `1 % -0` would come out as `-NaN`.
  if d.is_nan() {
    return D::NAN.with_ctx(CTX);
  }
  if !d.is_finite() {
    return d;
  }
  let digits = d.digits_count() as i32;
  let d = if digits > PRECISION {
    let scale = (d.fractional_digits_count() as i32 - (digits - PRECISION)).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    rescale_half_even(d, scale)
  } else {
    d
  };
  if d.is_zero() {
    return d;
  }
  if ilogb(d) > MAX_ADJUSTED_EXPONENT {
    // Above the format's ceiling: infinity, as the C library returns.
    if d.is_sign_negative() {
      D::NEG_INFINITY.with_ctx(CTX)
    } else {
      D::INFINITY.with_ctx(CTX)
    }
  } else if ilogb(d) < MIN_ADJUSTED_EXPONENT {
    // ...and below its floor: zero. fastnum's exponent range is far wider
    // (i16), so without this `1e-6000 * 1e-6000` keeps a value the
    // reference has already flushed away, and a decision comparing the
    // result to zero would answer differently on the two implementations.
    D::ZERO.with_ctx(CTX)
  } else if d.fractional_digits_count() < -MAX_STORED_EXPONENT {
    // In range, but not in this *cohort*: decimal128's exponent field
    // tops out at +6111, so a value above that is held with a padded
    // coefficient instead — the C library reads `1e6144` back as
    // `1000000000000000000000000000000000E+6111`. The same number, and
    // `Display` cannot tell them apart, but the stored scale is what
    // every later operation rounds against. The padding always fits:
    // `ilogb <= 6144` was just established, so widening the coefficient
    // to exponent 6111 leaves at most 34 digits.
    rescale_half_even(d, -MAX_STORED_EXPONENT)
  } else {
    d
  }
}

/// Parse without normalising — the one entry point that must not recurse into
/// [`norm`], which calls back here to rebuild a rounded value.
fn parse_raw(s: &str) -> D {
  match D::from_str(s, CTX) {
    Ok(d) => d,
    Err(_) => D::NAN.with_ctx(CTX),
  }
}

/// Parse, or NaN. Used only for internally-generated strings (integers and
/// powers of ten), which cannot fail; `FromStr` below reports a real error.
fn dec(s: &str) -> D {
  norm(parse_raw(s))
}

/// Round to `new_scale` decimal places, half-even — **not** fastnum's
/// `rescale`.
///
/// fastnum 0.7.5 drops the sticky bit under `RoundingMode::HalfEven`: it
/// decides from the first discarded digit alone, so a remainder of `5000001`
/// reads as an exact tie and `0.5000001` rounds to `0`, `2.5000001` to `2`.
/// Its `ceil`, `floor`, `trunc` and `HalfUp` are unaffected and stay
/// delegated; only this path is reimplemented, on the digit string, where the
/// whole remainder is visible. The differential in `feel-number-parity` is
/// what found this, and is what keeps it found.
fn rescale_half_even(d: D, new_scale: i16) -> D {
  if !d.is_finite() {
    return d;
  }
  let scale = d.fractional_digits_count();
  // Growing the scale only appends zeros — no decision to make, and
  // fastnum's own rescale preserves the value exactly.
  if new_scale >= scale {
    return d.rescale(new_scale);
  }
  let digits = d.digits().to_string();
  let drop = (scale as i32 - new_scale as i32) as usize;
  let (kept, dropped) = if drop >= digits.len() {
    ("0".to_string(), format!("{}{}", "0".repeat(drop - digits.len()), digits))
  } else {
    (digits[..digits.len() - drop].to_string(), digits[digits.len() - drop..].to_string())
  };
  let round_up = match compare_to_half(&dropped) {
    Ordering::Greater => true,
    Ordering::Less => false,
    // The tie: round to the even neighbour.
    Ordering::Equal => kept.as_bytes().last().is_some_and(|b| (b - b'0') % 2 == 1),
  };
  let kept = if round_up { increment(&kept) } else { kept };
  let sign = if d.is_sign_negative() { "-" } else { "" };
  let rounded = parse_raw(&format!("{sign}{kept}E{}", -new_scale));
  // A carry can widen the coefficient (999 -> 1000). One more pass settles
  // it, and that pass only ever discards trailing zeros, so it cannot round
  // twice: `99999...9` to 34 digits is `1000...0` at one higher exponent.
  if rounded.digits_count() as i32 > PRECISION {
    let scale = (rounded.fractional_digits_count() as i32 - (rounded.digits_count() as i32 - PRECISION)).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    rounded.rescale(scale)
  } else {
    rounded
  }
}

/// Compare a discarded digit string against one half of a unit in its last
/// kept place — i.e. against `5000…0` of the same width.
fn compare_to_half(dropped: &str) -> Ordering {
  let mut bytes = dropped.bytes();
  match bytes.next() {
    None => Ordering::Less,
    Some(b) if b > b'5' => Ordering::Greater,
    Some(b) if b < b'5' => Ordering::Less,
    // Leading 5: a tie only if nothing follows it. This is the sticky bit
    // fastnum loses.
    Some(_) => {
      if bytes.any(|b| b != b'0') {
        Ordering::Greater
      } else {
        Ordering::Equal
      }
    }
  }
}

/// Add one to a decimal digit string. The coefficient can be 77 digits wide,
/// which no primitive integer holds, so the carry is done on the text.
fn increment(digits: &str) -> String {
  let mut out = digits.as_bytes().to_vec();
  for i in (0..out.len()).rev() {
    if out[i] == b'9' {
      out[i] = b'0';
    } else {
      out[i] += 1;
      return String::from_utf8(out).expect("ascii digits");
    }
  }
  let mut carried = String::from("1");
  carried.push_str(&String::from_utf8(out).expect("ascii digits"));
  carried
}

/// `value × 10^n`, preserving the coefficient exactly — `bid128_scalbn`.
fn scalbn(d: D, n: i32) -> D {
  norm(d * dec(&format!("1E{n}")))
}

/// Whether a value is an even integer, decided from its own digits — or
/// `None` when the question does not arise (non-finite, or not an integer),
/// leaving the caller on its usual path.
///
/// The usual path is `remainder(2)`, and that rounds its quotient to 34
/// significant digits. For an integer wider than that, the last digit — the
/// only one parity depends on — is gone before the remainder is taken, so
/// `20000000000000000000000000000000050` came back odd. The C library's
/// remainder is exact there, and the differential says so.
fn exact_parity(d: D) -> Option<bool> {
  if !d.is_finite() {
    return None;
  }
  if d.is_zero() {
    return Some(true);
  }
  if !num_eq(d, trunc_i(d)) {
    return None;
  }
  let scale = i64::from(d.fractional_digits_count());
  if scale < 0 {
    // A negative scale is a trailing power of ten — even whatever the
    // coefficient is. Zero is *not* that case: there the coefficient is
    // the value, and its own last digit decides.
    return Some(true);
  }
  let digits = d.digits().to_string();
  let len = digits.len() as i64;
  // Integrality above means the last `scale` digits are zeros, so the units
  // digit sits `scale` places from the end.
  let units = usize::try_from(len - scale - 1).ok()?;
  Some(digits.as_bytes()[units].is_multiple_of(2))
}

/// Round to an integer, away from zero on a tie — `bid128_round_integral_nearest_away`.
fn nearest_away(d: D) -> D {
  // Same reason `floor_i` and friends have one: fastnum rounds through
  // `rescale`, which answers NaN for infinity where the C library returns
  // the infinity unchanged.
  if !d.is_finite() {
    return d;
  }
  d.with_rounding_mode(RoundingMode::HalfUp).round(0).with_ctx(CTX)
}

// ---------------------------------------------------------------- comparison
//
// fastnum orders `-0` *below* `0` while also reporting them equal, so its
// `<` and `==` disagree about signed zeros. The C library compares
// numerically (`bid128_quiet_less(-0, 0)` is false), and FEEL needs that: a
// `-0 < 0` that answers true would make `is_negative(-0)` true and take the
// wrong branch of a decision table. Every comparison below therefore settles
// the both-zero case first.

fn num_eq(a: D, b: D) -> bool {
  if a.is_zero() && b.is_zero() {
    return true;
  }
  a == b
}

fn num_lt(a: D, b: D) -> bool {
  !(a.is_zero() && b.is_zero()) && a < b
}

fn num_gt(a: D, b: D) -> bool {
  !(a.is_zero() && b.is_zero()) && a > b
}

fn num_le(a: D, b: D) -> bool {
  (a.is_zero() && b.is_zero()) || a <= b
}

fn num_ge(a: D, b: D) -> bool {
  (a.is_zero() && b.is_zero()) || a >= b
}

/// log10(e) — converts a natural exponent into decimal orders of magnitude.
const LOG10_E: f64 = std::f64::consts::LOG10_E;

/// `e ** x`, standing in for `bid128_exp`.
///
/// Same shape of problem as [`pow`], one step worse: for an argument far
/// outside decimal128's range fastnum does not merely panic, it **overflows
/// the stack** (`exp(1e6000)` aborts the process). Since every such argument
/// has infinity or zero as its answer, the range check settles them here.
fn exp(d: D) -> D {
  let magnitude = f64::from(d) * LOG10_E;
  if magnitude > MAX_ADJUSTED_EXPONENT as f64 {
    return D::INFINITY.with_ctx(CTX);
  }
  if magnitude < MIN_ADJUSTED_EXPONENT as f64 {
    return D::ZERO.with_ctx(CTX);
  }
  d.exp()
}

/// `base ** exponent`, standing in for `bid128_pow`.
///
/// Two things this cannot delegate to fastnum. **`x ** 0` is 1** for every
/// `x` per IEEE 754, which the C library follows and fastnum does not (it
/// answers NaN for `0 ** 0`). And fastnum's integer-exponent path overflows
/// its own scale and **panics** — `10 ** 39995` does, despite the traps being
/// off — so results whose magnitude leaves decimal128's range are settled
/// here instead of being computed. That costs nothing: outside that range the
/// answer is infinity or zero, which is what the C library returns too.
fn pow(base: D, exponent: D) -> D {
  if exponent.is_zero() {
    return D::ONE.with_ctx(CTX);
  }
  if base.is_zero() {
    return if exponent.is_sign_negative() {
      D::INFINITY.with_ctx(CTX)
    } else {
      D::ZERO.with_ctx(CTX)
    };
  }
  // A negative base raised to a non-integer power has no real value: NaN,
  // per IEEE 754 and the C library. fastnum answers as if the base were
  // positive, which would make `(-10) ** 0.5` a confident 3.16.
  if base.is_sign_negative() && !num_eq(exponent, exponent.trunc()) {
    return D::NAN.with_ctx(CTX);
  }
  // A negative base's sign is the parity of the exponent, and fastnum
  // decides it in a path that loses parity above `i32` — `(-1) ** 2147483648`
  // came out negative, and `(-1) ** 1e30` too. Settle the sign from the
  // exponent's own digits and raise the magnitude instead.
  if base.is_sign_negative()
    && let Some(even) = exact_parity(exponent)
  {
    let magnitude = pow(base.abs(), exponent);
    return if even { magnitude } else { -magnitude };
  }
  // An estimate is enough: it only decides whether the exact answer is
  // inside decimal128's range at all, and the guard sits far below where
  // fastnum starts panicking.
  let magnitude = log10_abs(base) * f64::from(exponent);
  if magnitude > MAX_ADJUSTED_EXPONENT as f64 {
    return D::INFINITY.with_ctx(CTX);
  }
  if magnitude < MIN_ADJUSTED_EXPONENT as f64 {
    return D::ZERO.with_ctx(CTX);
  }
  base.pow(exponent)
}

/// Round-to-integral, standing in for `bid128_round_integral_*`.
///
/// fastnum's `ceil`/`floor`/`trunc` go through `rescale`, which answers NaN
/// for a non-finite input; IEEE 754 and the C library return the value
/// unchanged. Without this `1e6000 % 1e-6000` — whose quotient overflows to
/// infinity — comes out NaN instead of -Infinity.
fn floor_i(d: D) -> D {
  if d.is_finite() { d.floor() } else { d }
}

fn ceil_i(d: D) -> D {
  if d.is_finite() { d.ceil() } else { d }
}

fn trunc_i(d: D) -> D {
  if d.is_finite() { d.trunc() } else { d }
}

/// Division, standing in for `bid128_div`.
///
/// `0 / 0` is NaN — invalid, per IEEE 754 — while `x / 0` for a non-zero `x`
/// is signed infinity. fastnum answers infinity for both, which would make
/// FEEL's `0 / 0` a value rather than the null it must be.
fn div(a: D, b: D) -> D {
  if a.is_zero() && b.is_zero() {
    return D::NAN.with_ctx(CTX);
  }
  a / b
}

/// Square root, standing in for `bid128_sqrt`.
///
/// fastnum's own `sqrt` overflows to infinity well inside decimal128's range
/// — `sqrt(1e600)` answers infinity where the value is a plain `1e300` — so
/// the exponent is halved first and put back afterwards:
/// `sqrt(x · 10^2k) = sqrt(x) · 10^k`. That keeps fastnum working near 1,
/// where it is reliable, and makes exact powers of ten come out exact instead
/// of as a 78-digit approximation.
fn sqrt(d: D) -> D {
  if d.is_zero() || !d.is_finite() || d.is_sign_negative() {
    return d.sqrt();
  }
  let half = ilogb(d).div_euclid(2);
  norm(scalbn(scalbn(d, -2 * half).sqrt(), half))
}

/// FEEL number.
#[derive(Copy, Clone)]
pub struct FeelNumber(D, bool);

// The second field is upstream's trailing-zero rule, and it is load-bearing
// for `Display`: `false` on values that came from a literal or a conversion
// (so `50.50` renders as typed), `true` on computed results (so a sum renders
// trimmed). Nothing else reads it.

impl FeelNumber {
  /// Returns [FeelNumber] value infinite.
  pub fn infinite() -> FeelNumber {
    Self(D::INFINITY.with_ctx(CTX), false)
  }

  /// Returns [FeelNumber] value 0 (zero).
  pub fn zero() -> FeelNumber {
    Self(D::ZERO.with_ctx(CTX), false)
  }

  /// Returns [FeelNumber] value 1 (one).
  pub fn one() -> FeelNumber {
    Self(D::ONE.with_ctx(CTX), false)
  }

  /// Returns [FeelNumber] value 2 (two).
  pub fn two() -> FeelNumber {
    Self(D::TWO.with_ctx(CTX), false)
  }

  /// Returns [FeelNumber] value 1_000_000_000 (billion).
  pub fn billion() -> FeelNumber {
    Self(dec("1000000000"), false)
  }

  /// Creates a new [FeelNumber] from integer value and a scale.
  ///
  /// The scale is the *stored* exponent, not a rounding instruction:
  /// `new(490, 1)` is `49.0` and renders with its trailing zero.
  pub fn new(n: i64, s: i32) -> Self {
    Self(dec(&format!("{n}E{}", -s)), false)
  }

  /// Creates a new [FeelNumber] from [isize].
  fn from_isize(n: isize) -> Self {
    Self(dec(&format!("{n}")), false)
  }

  /// Returns an absolute value of this [FeelNumber].
  pub fn abs(&self) -> Self {
    Self(self.0.abs(), false)
  }

  /// Returns a nearest integer greater or equal to this [FeelNumber].
  pub fn ceiling(&self, scale: i32) -> Result<Self> {
    Ok(Self(
      if scale == 0 {
        let rounded = ceil_i(self.0);
        // Without this, ceiling(-0.3333) renders as "-0".
        if rounded.is_zero() { D::ZERO.with_ctx(CTX) } else { rounded }
      } else {
        self.validate_scale(scale)?;
        self.unscale(ceil_i(scalbn(self.0, scale)), scale)
      },
      true,
    ))
  }

  pub fn even(&self) -> bool {
    // Deliberately no integrality check — upstream has none here, and
    // `odd` below has one. The asymmetry is upstream's.
    exact_parity(self.0).unwrap_or_else(|| self.remainder(D::TWO.with_ctx(CTX)).is_zero())
  }

  pub fn exp(&self) -> Option<Self> {
    let n = norm(exp(self.0));
    if n.is_finite() { Some(Self(n, true)) } else { None }
  }

  /// Returns a nearest integer less or equal to this [FeelNumber].
  pub fn floor(&self, scale: i32) -> Result<Self> {
    Ok(Self(
      if scale == 0 {
        floor_i(self.0)
      } else {
        self.validate_scale(scale)?;
        self.unscale(floor_i(scalbn(self.0, scale)), scale)
      },
      true,
    ))
  }

  pub fn frac(&self) -> Self {
    Self(norm(self.0 - trunc_i(self.0)), true)
  }

  pub fn is_integer(&self) -> bool {
    num_eq(self.0, trunc_i(self.0))
  }

  pub fn is_zero(&self) -> bool {
    self.0.is_zero()
  }

  pub fn is_one(&self) -> bool {
    num_eq(self.0, D::ONE.with_ctx(CTX))
  }

  pub fn is_negative(&self) -> bool {
    num_lt(self.0, D::ZERO.with_ctx(CTX))
  }

  pub fn is_positive(&self) -> bool {
    num_gt(self.0, D::ZERO.with_ctx(CTX))
  }

  pub fn ln(&self) -> Option<Self> {
    let n = norm(self.0.ln());
    if n.is_finite() { Some(Self(n, true)) } else { None }
  }

  pub fn odd(&self) -> bool {
    if self.is_integer() {
      exact_parity(self.0).map_or_else(|| !self.remainder(D::TWO.with_ctx(CTX)).is_zero(), |even| !even)
    } else {
      false
    }
  }

  pub fn pow(&self, rhs: &FeelNumber) -> Option<Self> {
    let n = norm(pow(self.0, rhs.0));
    if n.is_finite() { Some(Self(n, true)) } else { None }
  }

  pub fn round(&self, rhs: &FeelNumber) -> Self {
    // `bid128_quantize(self, 10^-scale, RND_NEAREST_EVEN)` upstream, which
    // is a half-even rescale — so it goes through our own rounding rather
    // than fastnum's `quantize`, which carries the same sticky-bit defect.
    let scale = to_i32_trunc(rhs.0).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    Self(norm(rescale_half_even(self.0, scale)), false)
  }

  pub fn round_down(&self, scale: i32) -> Result<Self> {
    Ok(Self(
      if scale == 0 {
        trunc_i(self.0)
      } else {
        self.validate_scale(scale)?;
        self.unscale(trunc_i(scalbn(self.0, scale)), scale)
      },
      true,
    ))
  }

  pub fn round_half_down(&self, scale: i32) -> Result<Self> {
    // Half-down is expressed as half-away-from-zero on a value nudged a
    // tenth toward zero, exactly as upstream does it.
    let tenth = dec("0.1");
    let positive = |n: D| nearest_away(norm(n - tenth));
    let negative = |n: D| nearest_away(norm(n + tenth));
    Ok(Self(
      if scale == 0 {
        if self.0.is_sign_negative() { negative(self.0) } else { positive(self.0) }
      } else {
        self.validate_scale(scale)?;
        let scaled = scalbn(self.0, scale);
        self.unscale(if self.0.is_sign_negative() { negative(scaled) } else { positive(scaled) }, scale)
      },
      true,
    ))
  }

  pub fn round_half_up(&self, scale: i32) -> Result<Self> {
    Ok(Self(
      if scale == 0 {
        nearest_away(self.0)
      } else {
        self.validate_scale(scale)?;
        self.unscale(nearest_away(scalbn(self.0, scale)), scale)
      },
      true,
    ))
  }

  pub fn round_up(&self, scale: i32) -> Result<Self> {
    let away = |n: D| {
      if self.0.is_sign_negative() { floor_i(n) } else { ceil_i(n) }
    };
    Ok(Self(
      if scale == 0 {
        away(self.0)
      } else {
        self.validate_scale(scale)?;
        self.unscale(away(scalbn(self.0, scale)), scale)
      },
      true,
    ))
  }

  pub fn sqrt(&self) -> Option<Self> {
    let n = norm(sqrt(self.0));
    if n.is_finite() { Some(Self(n, true)) } else { None }
  }

  pub fn square(&self) -> Option<Self> {
    let n = norm(pow(self.0, D::TWO.with_ctx(CTX)));
    if n.is_finite() { Some(Self(n, true)) } else { None }
  }

  pub fn trunc(&self) -> Self {
    Self(trunc_i(self.0), false)
  }

  /// Calculates the remainder of the division — **floored**, not truncated,
  /// which is what FEEL's `modulo` is: `-12 % 5` is `3`, not `-2`.
  fn remainder(&self, rhs: D) -> D {
    let n = floor_i(norm(div(self.0, rhs)));
    norm(self.0 - norm(rhs * n))
  }

  /// Validates effective scale after changing the scale to the required one.
  /// When the final scale is out of allowed range then returns an error.
  fn validate_scale(&self, scale: i32) -> Result<()> {
    // Saturating, not wrapping: `ilogb` answers `i32::MIN` for zero, and
    // upstream's plain `+` overflows there — a panic we decline to
    // reproduce. The refusal it produces for every in-range scale is
    // preserved, which is the part callers can observe.
    let scale = ilogb(self.0).saturating_add(scale);
    if (MIN_SCALE..=MAX_SCALE).contains(&scale) {
      Ok(())
    } else {
      Err(err_invalid_scale(MIN_SCALE, MAX_SCALE, scale))
    }
  }

  fn unscale(&self, n: D, scale: i32) -> D {
    if n.is_zero() {
      D::ZERO.with_ctx(CTX)
    } else if num_eq(n, D::ONE.with_ctx(CTX)) {
      D::ONE.with_ctx(CTX)
    } else if num_eq(n, -D::ONE.with_ctx(CTX)) {
      -D::ONE.with_ctx(CTX)
    } else {
      scalbn(n, -scale)
    }
  }
}

/// Truncating conversion to i32 — `bid128_to_int32_int`. Out of range or
/// non-finite yields zero, which is where upstream's undefined result lands
/// for the one caller (`round`, whose argument is a small scale).
fn to_i32_trunc(d: D) -> i32 {
  i32::try_from(d.trunc()).unwrap_or(0)
}

impl PartialEq<FeelNumber> for FeelNumber {
  fn eq(&self, rhs: &Self) -> bool {
    num_eq(self.0, rhs.0)
  }
}

impl PartialEq<FeelNumber> for isize {
  fn eq(&self, rhs: &FeelNumber) -> bool {
    num_eq(FeelNumber::from_isize(*self).0, rhs.0)
  }
}

impl PartialEq<isize> for FeelNumber {
  fn eq(&self, rhs: &isize) -> bool {
    num_eq(self.0, FeelNumber::from_isize(*rhs).0)
  }
}

impl PartialOrd<FeelNumber> for FeelNumber {
  /// Never `None`, even for NaN — upstream's three-way test, kept because a
  /// caller can observe the difference.
  fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
    if num_eq(self.0, rhs.0) {
      return Some(Ordering::Equal);
    }
    if num_gt(self.0, rhs.0) {
      return Some(Ordering::Greater);
    }
    Some(Ordering::Less)
  }

  fn lt(&self, rhs: &FeelNumber) -> bool {
    num_lt(self.0, rhs.0)
  }

  fn le(&self, rhs: &FeelNumber) -> bool {
    num_le(self.0, rhs.0)
  }

  fn gt(&self, rhs: &FeelNumber) -> bool {
    num_gt(self.0, rhs.0)
  }

  fn ge(&self, rhs: &FeelNumber) -> bool {
    num_ge(self.0, rhs.0)
  }
}

impl PartialOrd<FeelNumber> for isize {
  fn partial_cmp(&self, rhs: &FeelNumber) -> Option<Ordering> {
    let n = FeelNumber::from_isize(*self).0;
    if num_eq(n, rhs.0) {
      return Some(Ordering::Equal);
    }
    if num_gt(n, rhs.0) {
      return Some(Ordering::Greater);
    }
    Some(Ordering::Less)
  }
}

impl PartialOrd<isize> for FeelNumber {
  fn partial_cmp(&self, rhs: &isize) -> Option<Ordering> {
    let n = FeelNumber::from_isize(*rhs).0;
    if num_eq(self.0, n) {
      return Some(Ordering::Equal);
    }
    if num_gt(self.0, n) {
      return Some(Ordering::Greater);
    }
    Some(Ordering::Less)
  }
}

impl Add<FeelNumber> for FeelNumber {
  type Output = Self;
  fn add(self, rhs: Self) -> Self::Output {
    Self(norm(self.0 + rhs.0), true)
  }
}

impl AddAssign<FeelNumber> for FeelNumber {
  fn add_assign(&mut self, rhs: Self) {
    self.0 = norm(self.0 + rhs.0);
    self.1 = true;
  }
}

impl Sub<FeelNumber> for FeelNumber {
  type Output = Self;
  fn sub(self, rhs: Self) -> Self::Output {
    Self(norm(self.0 - rhs.0), true)
  }
}

impl SubAssign<FeelNumber> for FeelNumber {
  fn sub_assign(&mut self, rhs: Self) {
    self.0 = norm(self.0 - rhs.0);
    self.1 = true;
  }
}

impl Mul<FeelNumber> for FeelNumber {
  type Output = Self;
  fn mul(self, rhs: Self) -> Self::Output {
    Self(norm(self.0 * rhs.0), true)
  }
}

impl MulAssign<FeelNumber> for FeelNumber {
  fn mul_assign(&mut self, rhs: Self) {
    self.0 = norm(self.0 * rhs.0);
    self.1 = true;
  }
}

impl Div<FeelNumber> for FeelNumber {
  type Output = Self;
  fn div(self, rhs: Self) -> Self::Output {
    Self(norm(div(self.0, rhs.0)), true)
  }
}

impl DivAssign<FeelNumber> for FeelNumber {
  fn div_assign(&mut self, rhs: Self) {
    self.0 = norm(div(self.0, rhs.0));
    self.1 = true;
  }
}

impl Rem<FeelNumber> for FeelNumber {
  type Output = Self;
  fn rem(self, rhs: Self) -> Self::Output {
    Self(self.remainder(rhs.0), true)
  }
}

impl RemAssign<FeelNumber> for FeelNumber {
  fn rem_assign(&mut self, rhs: Self) {
    self.0 = self.remainder(rhs.0);
    self.1 = true;
  }
}

impl Neg for FeelNumber {
  type Output = Self;
  fn neg(self) -> Self::Output {
    Self(self.0.neg(), true)
  }
}

impl Debug for FeelNumber {
  /// The C library's canonical form, which upstream's `Debug` inherits
  /// verbatim: sign, coefficient as stored, `E`, signed exponent.
  /// `FeelNumber::new(49, 0)` is `+49E+0`.
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if !self.0.is_finite() {
      // `bid128_to_string`'s forms, which upstream's Debug inherits:
      // `+Inf`, `-Inf`, `+NaN`. Unlike Display, this path does not
      // panic upstream, so it must match exactly.
      return write!(f, "{}{}", if self.0.is_sign_negative() { "-" } else { "+" }, if self.0.is_nan() { "NaN" } else { "Inf" });
    }
    let exponent = -(self.0.fractional_digits_count() as i32);
    write!(
      f,
      "{}{}E{}{}",
      if self.0.is_sign_negative() { "-" } else { "+" },
      self.0.digits(),
      if exponent < 0 { "-" } else { "+" },
      exponent.abs()
    )
  }
}

impl Display for FeelNumber {
  /// Converts [FeelNumber] to human readable string.
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if !self.0.is_finite() {
      // Upstream panics here; see the crate docs.
      return f.pad(non_finite(self.0));
    }
    let negative = self.0.is_sign_negative();
    let sb = self.0.digits().to_string();
    let exponent = -(self.0.fractional_digits_count() as isize);
    let decimal_points = exponent.unsigned_abs();
    let (mut before, mut after) = if exponent < 0 {
      let digit_count = sb.len();
      if digit_count <= decimal_points {
        let before = "0".to_string();
        let mut after = "0".repeat(decimal_points - digit_count);
        if self.1 {
          after.push_str(sb.trim_end_matches('0'));
        } else {
          after.push_str(&sb);
        }
        (before, after)
      } else {
        let before = sb[..digit_count - decimal_points].to_string();
        let after = if self.1 {
          sb[digit_count - decimal_points..].trim_end_matches('0').to_string()
        } else {
          sb[digit_count - decimal_points..].to_string()
        };
        (before, after)
      }
    } else {
      let mut before = sb.to_string();
      before.push_str(&"0".repeat(decimal_points));
      let after = "".to_string();
      (before, after)
    };
    if let Some(precision) = f.precision() {
      if after.len() < precision {
        after.push_str(&"0".repeat(precision - after.len()));
      } else {
        after = after[0..precision].to_string();
      }
    }
    if !after.is_empty() {
      before.push('.');
      before.push_str(&after);
    }
    f.pad_integral(!negative, "", &before)
  }
}

/// The three values the C library renders and upstream's `Display` cannot.
fn non_finite(d: D) -> &'static str {
  if d.is_nan() {
    "NaN"
  } else if d.is_sign_negative() {
    "-Infinity"
  } else {
    "Infinity"
  }
}

impl Jsonify for FeelNumber {
  /// Converts [FeelNumber] to JSON string.
  fn jsonify(&self) -> String {
    format!("{self}")
  }
}

impl FromStr for FeelNumber {
  type Err = DsntkError;
  fn from_str(s: &str) -> Result<Self, Self::Err> {
    // Leading **space and tab**, and nothing else — measured, not assumed.
    //
    // Not `trim_start()` and not C's `isspace`: `bid128_from_string`
    // accepts `" 1"` and `"\t1"` and refuses `"\n1"`, `"\r1"`, `"\x0b1"`
    // and `"\x0c1"`. Two readings of this got it wrong in opposite
    // directions — the first set included `\n` and `\r`, which the C
    // library rejects; a review then proposed the whole `isspace` set,
    // which is wider still. The corpus now carries all six characters, so
    // the next reading does not have to guess either.
    match D::from_str(s.trim_start_matches([' ', '\t']), CTX) {
      // Finiteness is checked *after* `norm`, not before. fastnum's type
      // is far wider than decimal128, so `1e6145` parses perfectly
      // finite and only overflows when clamped into range — checking
      // first handed back `Infinity` for a literal the C library
      // refuses, and every later operation inherited it.
      Ok(n) => {
        let n = norm(n);
        if n.is_finite() { Ok(Self(n, false)) } else { Err(err_invalid_number_literal(s)) }
      }
      Err(_) => Err(err_invalid_number_literal(s)),
    }
  }
}

macro_rules! from_int {
    ($($t:ty),*) => {
        $(impl From<$t> for FeelNumber {
            fn from(value: $t) -> Self {
                Self(dec(&format!("{value}")), false)
            }
        })*
    };
}

from_int!(u8, i8, u16, i16, u32, i32, u64, i64, isize, usize);

/// Truncating, like the C library's `bid128_to_*_int`: `0.999` converts to
/// `0`, and only being out of range or non-finite is an error.
macro_rules! try_into_int {
    ($($t:ty),*) => {
        $(
            impl TryFrom<FeelNumber> for $t {
                type Error = DsntkError;
                fn try_from(value: FeelNumber) -> Result<Self, Self::Error> {
                    <$t>::try_from(&value)
                }
            }

            impl TryFrom<&FeelNumber> for $t {
                type Error = DsntkError;
                fn try_from(value: &FeelNumber) -> Result<Self, Self::Error> {
                    if !value.0.is_finite() {
                        return Err(err_number_conversion_failed());
                    }
                    let truncated = trunc_i(value.0);
                    let wide = i128::try_from(truncated).map_err(|_| err_number_conversion_failed())?;
                    <$t>::try_from(wide).map_err(|_| err_number_conversion_failed())
                }
            }
        )*
    };
}

try_into_int!(u8, i8, u16, i16, u32, i32, u64, i64, usize, isize);
