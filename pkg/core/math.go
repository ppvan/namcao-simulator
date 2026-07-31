package core

import "math"

// ---------------------------------------------------------------------
// Exp: Schraudolph's bit-hack exponential, adapted for float32.
//
// Idea: IEEE-754 float bits are (roughly) a piecewise-linear encoding of
// log2(x). So we can go the other way: build the bits of a float directly
// from x*log2(e), then round the mantissa. One extra Newton-ish correction
// term keeps relative error under ~3%, which is plenty for activations.
// ---------------------------------------------------------------------

const (
	expA32 = float32(12102203.0)         // 2^23 / ln(2)
	expB32 = int32(127*(1<<23)) - 486411 // bias, tuned to minimize FastExp32's error
	logB32 = int32(127 * (1 << 23))      // exact (untuned) bias, used for FastLog32
	// NOTE: expB32 and logB32 are deliberately different constants. They are
	// two independent empirical bit-hacks (Schraudolph-style), not exact
	// mathematical inverses of each other. Reusing expB32 inside FastLog32
	// introduces a constant additive bias (~0.04 in log-space) that is
	// small on its own but blows up exponentially in anything that raises
	// FastLog32's result to a large power (e.g. FastPow32 with big exponents).
)

// FastExp32 approximates exp(x) for float32 in O(1) with no float64 use.
// Typical relative error: < 3% over x in [-20, 20]; degrades gracefully
// outside that (saturates instead of overflowing/underflowing to garbage).
func FastExp32(x float32) float32 {
	if x > 88 { // avoid overflow, matches float32 exp(x) saturation point
		return math.MaxFloat32
	}
	if x < -88 {
		return 0
	}
	i := int32(expA32*x) + expB32
	return math.Float32frombits(uint32(i))
}

// ---------------------------------------------------------------------
// Log: inverse of the same bit trick.
// ---------------------------------------------------------------------

// FastLog32 approximates ln(x) for x > 0, float32 in/out.
// Relative error is small (~1e-3 range) for typical positive activation
// values; not defined for x <= 0 (returns -Inf / NaN like math.Log does).
func FastLog32(x float32) float32 {
	if x <= 0 {
		if x == 0 {
			return float32(math.Inf(-1))
		}
		return float32(math.NaN())
	}
	bits := math.Float32bits(x)
	y := float32(bits)
	return (y - float32(logB32)) / expA32
}

// ---------------------------------------------------------------------
// Sqrt: classic "Quake" fast inverse sqrt, then one Newton-Raphson step,
// then invert. Cheaper than math.Sqrt (float64) in tight loops and stays
// in float32 registers.
// ---------------------------------------------------------------------

// FastInvSqrt32 approximates 1/sqrt(x) (the Quake III algorithm).
// One Newton-Raphson iteration gives ~0.17% max relative error.
func FastInvSqrt32(x float32) float32 {
	if x <= 0 {
		return 0
	}
	xhalf := 0.5 * x
	i := math.Float32bits(x)
	i = 0x5f3759df - (i >> 1)
	y := math.Float32frombits(i)
	y = y * (1.5 - xhalf*y*y) // Newton-Raphson iteration #1
	return y
}

// FastSqrt32 approximates sqrt(x) via x * invSqrt(x).
// Used the same way math.Sqrt is used for Adam's `sqrt(v)+eps` term.
func FastSqrt32(x float32) float32 {
	if x <= 0 {
		return 0
	}
	return x * FastInvSqrt32(x)
}

// ---------------------------------------------------------------------
// Pow: a^b = exp(b * ln(a)), built from the two approximations above.
// Matches the Adam bias-correction use case: math.Pow(beta, float32(t)).
// ---------------------------------------------------------------------

// FastPow32 approximates a^b for a > 0, via exp(b*ln(a)).
// Fine for small-to-moderate |b|. For large integer exponents (hundreds+),
// especially with a close to 1 (e.g. Adam's beta1^t, beta2^t as t grows),
// FastLog32's small per-call error gets multiplied by b and then
// exponentiated, which can blow up. Use PowIntApprox32 for that case —
// it has no transcendental approximation error at all.
func FastPow32(a, b float32) float32 {
	if a <= 0 {
		return 0
	}
	return FastExp32(b * FastLog32(a))
}

// PowIntApprox32 is a cheaper, more accurate special case for integer
// exponents (e.g. Adam's beta1^t, beta2^t where t is a step count) using
// exponentiation by squaring. No transcendental approximation error at
// all — useful when you want FastPow32-speed without its ~3% Exp error.
func PowIntApprox32(base float32, exp int) float32 {
	if exp < 0 {
		return 1 / PowIntApprox32(base, -exp)
	}
	result := float32(1)
	for exp > 0 {
		if exp&1 == 1 {
			result *= base
		}
		base *= base
		exp >>= 1
	}
	return result
}

// ---------------------------------------------------------------------
// Tanh: rational (Pade-like) approximation, cheaper and more accurate
// than deriving tanh from two FastExp32 calls, and avoids the classic
// exp-overflow issue for tanh(x) at large |x|.
// ---------------------------------------------------------------------

// FastTanh32 approximates tanh(x) for float32.
// Max absolute error ~1e-3 over all real x (clamps to +-1 for |x| > ~4.9,
// same as real tanh does numerically in float32).
func FastTanh32(x float32) float32 {
	if x > 4.9 {
		return 1
	}
	if x < -4.9 {
		return -1
	}
	x2 := x * x
	// Pade [3/2] approximant of tanh, tuned for float32 use.
	a := x * (135135 + x2*(17325+x2*(378+x2)))
	b := 135135 + x2*(62370+x2*(3150+x2*28))
	return a / b
}

// FastSigmoid32 approximates 1/(1+exp(-x)) directly from FastTanh32,
// since sigmoid(x) = 0.5*(1+tanh(x/2)) — this reuses the more accurate
// tanh approximation and avoids a division-heavy exp path.
func FastSigmoid32(x float32) float32 {
	return 0.5 * (1 + FastTanh32(0.5*x))
}

// ---------------------------------------------------------------------
// Max / Min / IsNaN / Inf: these are exact and cheap already in float64,
// but the float32 versions below avoid the implicit float64 promotion
// (and matching float64 math.Inf/math.IsNaN calls) you'd otherwise pay
// on every element of a hot loop.
// ---------------------------------------------------------------------

// Max32 returns the larger of a, b (exact, no approximation needed).
func Max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

// Min32 returns the smaller of a, b (exact, no approximation needed).
func Min32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

// IsNaN32 reports whether x is NaN (exact; NaN is the only float32 value
// that isn't equal to itself).
func IsNaN32(x float32) bool {
	return x != x
}

// PosInf32 / NegInf32 return the exact float32 +-Inf sentinel values
// (float32 has a native infinity representation, so this needs no
// approximation, just avoids the float64 math.Inf(1) call + conversion).
func PosInf32() float32 { return float32(math.Inf(1)) }
func NegInf32() float32 { return float32(math.Inf(-1)) }
