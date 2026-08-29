package core

import "math"

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

func FastExp32(x float32) float32 {
	if x > 88 { // avoid overflow, matches float32 exp(x) saturation point
		return math.MaxFloat32
	}
	if x < -88 {
		return 0
	}
	i := int32(expA32*x) + expB32
	if i <= 0 {
		// The -88 guard above is slightly too generous: the bit pattern goes
		// negative at x ~= -87.988, and uint32(negative) is a NaN/-Inf
		// encoding. Underflow to zero instead.
		return 0
	}
	return math.Float32frombits(uint32(i))
}

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

func FastSqrt32(x float32) float32 {
	if x <= 0 {
		return 0
	}
	return x * FastInvSqrt32(x)
}

func FastPow32(a, b float32) float32 {
	if a <= 0 {
		return 0
	}
	return FastExp32(b * FastLog32(a))
}

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

func FastSigmoid32(x float32) float32 {
	return 0.5 * (1 + FastTanh32(0.5*x))
}

func Max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func Min32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func IsNaN32(x float32) bool {
	return x != x
}

func PosInf32() float32 { return float32(math.Inf(1)) }
func NegInf32() float32 { return float32(math.Inf(-1)) }

func FastDSigmoid32(y float32) float32 {
	return y * (1 - y)
}

func FastDTanh32(y float32) float32 {
	return 1 - y*y
}
