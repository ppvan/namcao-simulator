package core

import (
	"math"
	"math/rand"
	"testing"
)

func ref64(a, b []float32, m, n, p int) []float64 {
	c := make([]float64, m*p)
	for i := 0; i < m; i++ {
		for k := 0; k < n; k++ {
			aik := float64(a[i*n+k])
			for j := 0; j < p; j++ {
				c[i*p+j] += aik * float64(b[k*p+j])
			}
		}
	}
	return c
}
func rms(x []float32, r []float64) float64 {
	var s, d float64
	for i := range x {
		e := float64(x[i]) - r[i]
		s += e * e
		d += r[i] * r[i]
	}
	return math.Sqrt(s / math.Max(d, 1e-30))
}

func refT64(a, b []float32, m, n, q int) []float64 {
	c := make([]float64, m*q)
	for i := 0; i < m; i++ {
		for j := 0; j < q; j++ {
			var s float64
			for k := 0; k < n; k++ {
				s += float64(a[i*n+k]) * float64(b[j*n+k])
			}
			c[i*q+j] = s
		}
	}
	return c
}

func refTDotAcc64(a, b, base []float32, m, n, p int) []float64 {
	c := make([]float64, n*p)
	for i := range base {
		c[i] = float64(base[i])
	}
	for k := 0; k < m; k++ {
		for i := 0; i < n; i++ {
			aki := float64(a[k*n+i])
			for j := 0; j < p; j++ {
				c[i*p+j] += aki * float64(b[k*p+j])
			}
		}
	}
	return c
}

func rnd(n int, r *rand.Rand) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = r.Float32()*2 - 1
	}
	return s
}
func maxrel(x, y []float32) float64 {
	var w float64
	for i := range x {
		d := math.Abs(float64(x[i] - y[i]))
		den := math.Max(math.Abs(float64(x[i])), 1e-3)
		if e := d / den; e > w {
			w = e
		}
	}
	return w
}

// shapes: real LSTM ones plus adversarial sizes for the tail paths
var shapesA = [][3]int{
	{1, 1024, 256}, {1, 1024, 32}, {1, 256, 65}, {1, 32, 1024}, // real
	{1, 1, 1}, {1, 7, 3}, {2, 9, 5}, {3, 8, 8}, {4, 17, 13}, {5, 1, 1},
	{7, 33, 65}, {1, 255, 255}, {2, 100, 3}, {1, 8, 4}, {1, 9, 4}, {1, 15, 7}, {6, 16, 16},
}
var shapesB = [][3]int{
	{1, 256, 1024}, {1, 32, 1024}, {1, 256, 65}, {64, 256, 1024}, {16, 256, 1024},
	{1, 1, 1}, {1, 3, 7}, {5, 8, 8}, {3, 9, 5}, {6, 17, 13}, {9, 33, 65}, {2, 255, 255},
	{17, 4, 100}, {4, 8, 8}, {4, 8, 9}, {5, 7, 15}, {8, 16, 16}, {3, 5, 1},
}

func TestDotTIntoMatchesScalar(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for _, s := range shapesA {
		m, n, q := s[0], s[1], s[2]
		a, b := rnd(m*n, r), rnd(q*n, r)
		c1, c2 := make([]float32, m*q), make([]float32, m*q)
		dotTIntoScalar(a, b, c1, m, n, q)
		dotTInto(a, b, c2, m, n, q)
		ref := refT64(a, b, m, n, q)
		e1, e2 := rms(c1, ref), rms(c2, ref)
		if e2 > 1e-5 || e2 > 6*e1+1e-9 {
			t.Errorf("dotTInto(%d,%d,%d): scalar rms %g, simd rms %g", m, n, q, e1, e2)
		}
	}
}

func TestTDotAccIntoMatchesScalar(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	for _, s := range shapesB {
		m, n, p := s[0], s[1], s[2]
		a, b, base := rnd(m*n, r), rnd(m*p, r), rnd(n*p, r)
		c1 := append([]float32(nil), base...)
		c2 := append([]float32(nil), base...)
		tDotAccIntoScalar(a, b, c1, m, n, p)
		tDotAccInto(a, b, c2, m, n, p)
		ref := refTDotAcc64(a, b, base, m, n, p)
		e1, e2 := rms(c1, ref), rms(c2, ref)
		if e2 > 1e-5 || e2 > 6*e1+1e-9 {
			t.Errorf("tDotAccInto(%d,%d,%d): scalar rms %g, simd rms %g", m, n, p, e1, e2)
		}
	}
}

// exercises the column-window path used by the parallel splitter
func TestDotTiledMatchesScalar(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	for _, s := range [][3]int{{1, 32, 1024}, {1, 256, 1024}, {1, 256, 65}, {8, 64, 64}, {3, 7, 13}, {5, 9, 17}, {1, 1, 1}, {40, 33, 71}} {
		m, n, p := s[0], s[1], s[2]
		a, b := rnd(m*n, r), rnd(n*p, r)
		// full window
		c1, c2 := make([]float32, m*p), make([]float32, m*p)
		dotTiledScalar(a, b, c1, 0, m, n, p, 0, p)
		dotTiled(a, b, c2, 0, m, n, p, 0, p)
		ref := ref64(a, b, m, n, p)
		e1, e2 := rms(c1, ref), rms(c2, ref)
		if e2 > 1e-5 || e2 > 6*e1+1e-9 {
			t.Errorf("dotTiled full (%d,%d,%d): scalar rms %g, simd rms %g", m, n, p, e1, e2)
		}
		// split column windows, mimicking dotInto's goroutine chunks
		c3, c4 := make([]float32, m*p), make([]float32, m*p)
		for cs := 0; cs < p; cs += 8 {
			ce := cs + 8
			if ce > p {
				ce = p
			}
			dotTiledScalar(a, b, c3, 0, m, n, p, cs, ce)
			dotTiled(a, b, c4, 0, m, n, p, cs, ce)
		}
		refw := ref64(a, b, m, n, p)
		f1, f2 := rms(c3, refw), rms(c4, refw)
		if f2 > 1e-5 || f2 > 6*f1+1e-9 {
			t.Errorf("dotTiled windowed (%d,%d,%d): scalar rms %g, simd rms %g", m, n, p, f1, f2)
		}
	}
}

// t=0 has hPrev == 0; the blocked path must not skip it incorrectly
func TestZeroRows(t *testing.T) {
	r := rand.New(rand.NewSource(4))
	for _, m := range []int{1, 4, 7, 64} {
		n, p := 256, 1024
		a := make([]float32, m*n)
		b, base := rnd(m*p, r), rnd(n*p, r)
		c1 := append([]float32(nil), base...)
		c2 := append([]float32(nil), base...)
		tDotAccIntoScalar(a, b, c1, m, n, p)
		tDotAccInto(a, b, c2, m, n, p)
		if e := maxrel(c1, c2); e != 0 {
			t.Errorf("zero rows m=%d: %g", m, e)
		}
	}
}

// no write may land outside the destination window
func TestNoOverrun(t *testing.T) {
	r := rand.New(rand.NewSource(5))
	const guard = 16
	for _, s := range [][3]int{{1, 256, 1023}, {1, 7, 3}, {3, 9, 5}, {1, 32, 1021}, {5, 17, 13}} {
		m, n, p := s[0], s[1], s[2]
		a, b := rnd(m*n, r), rnd(m*p, r)
		buf := make([]float32, n*p+guard)
		for i := n * p; i < len(buf); i++ {
			buf[i] = 12345
		}
		tDotAccInto(a, b, buf[:n*p], m, n, p)
		for i := n * p; i < len(buf); i++ {
			if buf[i] != 12345 {
				t.Fatalf("tDotAccInto overran at +%d (shape %v)", i-n*p, s)
			}
		}
		ab := rnd(s[2]*n, r)
		out := make([]float32, m*p+guard)
		for i := m * p; i < len(out); i++ {
			out[i] = 12345
		}
		dotTInto(a, ab, out[:m*p], m, n, p)
		for i := m * p; i < len(out); i++ {
			if out[i] != 12345 {
				t.Fatalf("dotTInto overran at +%d (shape %v)", i-m*p, s)
			}
		}
	}
}
