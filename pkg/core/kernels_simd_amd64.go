//go:build goexperiment.simd && amd64

package core

import "simd/archsimd"

// simdOK reports whether this CPU has the instructions these kernels compile
// down to (VFMADD213PS + 256-bit loads).
//
// archsimd intrinsics PANIC on unsupported hardware — there is no automatic
// fallback and, as of Go 1.26, no vet check that catches a missing guard — so
// every entry point below tests this once and routes to the scalar kernel
// otherwise.
//
// Float32x8 (AVX2) is the right width to target for Ryzen: Zen 3 (5000
// series) has AVX2+FMA but no AVX-512, Zen 4/5 (7000/9000) have both but
// execute 512-bit ops as two 256-bit halves on Zen 4. Float32x16 is a
// mechanical substitution if you want to try it on Zen 5.
var simdOK = archsimd.X86.AVX2() && archsimd.X86.FMA()

const simdLanes = 8

// dotTInto computes C(m,q) = A(m,n) · B(q,n)ᵀ.
//
// The reduction runs along n, which is the contiguous axis of both operands,
// so the vector accumulator sits in a register for the whole dot product and
// only the horizontal sum touches memory.
func dotTInto(a, b, c []float32, m, n, q int) {
	if !simdOK {
		dotTIntoScalar(a, b, c, m, n, q)
		return
	}
	var lanes [simdLanes]float32
	for i := 0; i < m; i++ {
		arow := a[i*n : i*n+n]
		crow := c[i*q : i*q+q]
		j := 0
		// Four output columns at a time. Two reasons: arow is loaded once
		// per four dot products instead of once per dot product, and four
		// independent accumulator chains hide the ~4 cycle FMA latency. A
		// single chain would run at one FMA per 4 cycles instead of two per
		// cycle, which throws away most of the win.
		for ; j+3 < q; j += 4 {
			b0 := b[(j+0)*n : (j+0)*n+n]
			b1 := b[(j+1)*n : (j+1)*n+n]
			b2 := b[(j+2)*n : (j+2)*n+n]
			b3 := b[(j+3)*n : (j+3)*n+n]
			var s0, s1, s2, s3 archsimd.Float32x8
			k := 0
			for ; k+simdLanes <= n; k += simdLanes {
				av := archsimd.LoadFloat32x8Slice(arow[k:])
				s0 = av.MulAdd(archsimd.LoadFloat32x8Slice(b0[k:]), s0)
				s1 = av.MulAdd(archsimd.LoadFloat32x8Slice(b1[k:]), s1)
				s2 = av.MulAdd(archsimd.LoadFloat32x8Slice(b2[k:]), s2)
				s3 = av.MulAdd(archsimd.LoadFloat32x8Slice(b3[k:]), s3)
			}
			if k < n {
				// Part loads zero-fill the lanes past the end, and
				// 0*x + acc == acc, so the tail needs no scalar path.
				av := archsimd.LoadFloat32x8SlicePart(arow[k:])
				s0 = av.MulAdd(archsimd.LoadFloat32x8SlicePart(b0[k:]), s0)
				s1 = av.MulAdd(archsimd.LoadFloat32x8SlicePart(b1[k:]), s1)
				s2 = av.MulAdd(archsimd.LoadFloat32x8SlicePart(b2[k:]), s2)
				s3 = av.MulAdd(archsimd.LoadFloat32x8SlicePart(b3[k:]), s3)
			}
			crow[j+0] = hsum8(s0, &lanes)
			crow[j+1] = hsum8(s1, &lanes)
			crow[j+2] = hsum8(s2, &lanes)
			crow[j+3] = hsum8(s3, &lanes)
		}
		for ; j < q; j++ {
			brow := b[j*n : j*n+n]
			var s archsimd.Float32x8
			k := 0
			for ; k+simdLanes <= n; k += simdLanes {
				s = archsimd.LoadFloat32x8Slice(arow[k:]).
					MulAdd(archsimd.LoadFloat32x8Slice(brow[k:]), s)
			}
			if k < n {
				s = archsimd.LoadFloat32x8SlicePart(arow[k:]).
					MulAdd(archsimd.LoadFloat32x8SlicePart(brow[k:]), s)
			}
			crow[j] = hsum8(s, &lanes)
		}
	}
}

// hsum8 adds the eight lanes of v. archsimd exposes no horizontal reduce, so
// this round-trips through the stack. It runs once per output element, i.e.
// once per n/8 FMAs, so it does not show up in the profile at n = 1024.
func hsum8(v archsimd.Float32x8, lanes *[simdLanes]float32) float32 {
	v.StoreSlice(lanes[:])
	return ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
		((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
}

// tDotAccInto computes C(n,p) += A(m,n)ᵀ · B(m,p).
//
// Loop order is i-then-k, not k-then-i as in the scalar version. That matters
// once you batch: with k outermost every batch row re-streams the whole (n,p)
// accumulator, so traffic grows linearly with the batch and batching buys
// nothing. Blocking k by 4 touches the accumulator m/4 times instead of m.
// At batch 1 the blocked loop never executes and the tail below is a plain
// vectorised axpy, identical in shape to the scalar version.
func tDotAccInto(a, b, c []float32, m, n, p int) {
	if !simdOK {
		tDotAccIntoScalar(a, b, c, m, n, p)
		return
	}
	for i := 0; i < n; i++ {
		crow := c[i*p : i*p+p]
		k := 0
		for ; k+3 < m; k += 4 {
			a0 := archsimd.BroadcastFloat32x8(a[(k+0)*n+i])
			a1 := archsimd.BroadcastFloat32x8(a[(k+1)*n+i])
			a2 := archsimd.BroadcastFloat32x8(a[(k+2)*n+i])
			a3 := archsimd.BroadcastFloat32x8(a[(k+3)*n+i])
			b0 := b[(k+0)*p : (k+0)*p+p]
			b1 := b[(k+1)*p : (k+1)*p+p]
			b2 := b[(k+2)*p : (k+2)*p+p]
			b3 := b[(k+3)*p : (k+3)*p+p]
			j := 0
			for ; j+simdLanes <= p; j += simdLanes {
				cv := archsimd.LoadFloat32x8Slice(crow[j:])
				cv = a0.MulAdd(archsimd.LoadFloat32x8Slice(b0[j:]), cv)
				cv = a1.MulAdd(archsimd.LoadFloat32x8Slice(b1[j:]), cv)
				cv = a2.MulAdd(archsimd.LoadFloat32x8Slice(b2[j:]), cv)
				cv = a3.MulAdd(archsimd.LoadFloat32x8Slice(b3[j:]), cv)
				cv.StoreSlice(crow[j:])
			}
			if j < p {
				cv := archsimd.LoadFloat32x8SlicePart(crow[j:])
				cv = a0.MulAdd(archsimd.LoadFloat32x8SlicePart(b0[j:]), cv)
				cv = a1.MulAdd(archsimd.LoadFloat32x8SlicePart(b1[j:]), cv)
				cv = a2.MulAdd(archsimd.LoadFloat32x8SlicePart(b2[j:]), cv)
				cv = a3.MulAdd(archsimd.LoadFloat32x8SlicePart(b3[j:]), cv)
				cv.StoreSlicePart(crow[j:])
			}
		}
		for ; k < m; k++ {
			aki := a[k*n+i]
			if aki == 0 {
				continue // hPrev is all zeros at t = 0
			}
			av := archsimd.BroadcastFloat32x8(aki)
			brow := b[k*p : k*p+p]
			j := 0
			for ; j+simdLanes <= p; j += simdLanes {
				cv := archsimd.LoadFloat32x8Slice(crow[j:])
				av.MulAdd(archsimd.LoadFloat32x8Slice(brow[j:]), cv).StoreSlice(crow[j:])
			}
			if j < p {
				cv := archsimd.LoadFloat32x8SlicePart(crow[j:])
				av.MulAdd(archsimd.LoadFloat32x8SlicePart(brow[j:]), cv).StoreSlicePart(crow[j:])
			}
		}
	}
}

// dotTiled computes result[rowStart:rowEnd, colStart:colEnd] += A · B.
// The tiling and the caller-supplied row/column window are unchanged; only
// the innermost axpy is vectorised.
func dotTiled(a, b, result []float32, rowStart, rowEnd, n, p, colStart, colEnd int) {
	if !simdOK {
		dotTiledScalar(a, b, result, rowStart, rowEnd, n, p, colStart, colEnd)
		return
	}
	width := colEnd - colStart
	if width <= 0 {
		return
	}
	for ii := rowStart; ii < rowEnd; ii += tileM {
		iMax := min(ii+tileM, rowEnd)
		for kk := 0; kk < n; kk += tileK {
			kMax := min(kk+tileK, n)
			for i := ii; i < iMax; i++ {
				aRow := i * n
				c := result[i*p+colStart : i*p+colEnd]

				k := kk
				for ; k+3 < kMax; k += 4 {
					a0 := archsimd.BroadcastFloat32x8(a[aRow+k])
					a1 := archsimd.BroadcastFloat32x8(a[aRow+k+1])
					a2 := archsimd.BroadcastFloat32x8(a[aRow+k+2])
					a3 := archsimd.BroadcastFloat32x8(a[aRow+k+3])
					b0 := b[(k+0)*p+colStart : (k+0)*p+colEnd]
					b1 := b[(k+1)*p+colStart : (k+1)*p+colEnd]
					b2 := b[(k+2)*p+colStart : (k+2)*p+colEnd]
					b3 := b[(k+3)*p+colStart : (k+3)*p+colEnd]
					j := 0
					for ; j+simdLanes <= width; j += simdLanes {
						cv := archsimd.LoadFloat32x8Slice(c[j:])
						cv = a0.MulAdd(archsimd.LoadFloat32x8Slice(b0[j:]), cv)
						cv = a1.MulAdd(archsimd.LoadFloat32x8Slice(b1[j:]), cv)
						cv = a2.MulAdd(archsimd.LoadFloat32x8Slice(b2[j:]), cv)
						cv = a3.MulAdd(archsimd.LoadFloat32x8Slice(b3[j:]), cv)
						cv.StoreSlice(c[j:])
					}
					if j < width {
						cv := archsimd.LoadFloat32x8SlicePart(c[j:])
						cv = a0.MulAdd(archsimd.LoadFloat32x8SlicePart(b0[j:]), cv)
						cv = a1.MulAdd(archsimd.LoadFloat32x8SlicePart(b1[j:]), cv)
						cv = a2.MulAdd(archsimd.LoadFloat32x8SlicePart(b2[j:]), cv)
						cv = a3.MulAdd(archsimd.LoadFloat32x8SlicePart(b3[j:]), cv)
						cv.StoreSlicePart(c[j:])
					}
				}
				for ; k < kMax; k++ {
					av := archsimd.BroadcastFloat32x8(a[aRow+k])
					bb := b[k*p+colStart : k*p+colEnd]
					j := 0
					for ; j+simdLanes <= width; j += simdLanes {
						cv := archsimd.LoadFloat32x8Slice(c[j:])
						av.MulAdd(archsimd.LoadFloat32x8Slice(bb[j:]), cv).StoreSlice(c[j:])
					}
					if j < width {
						cv := archsimd.LoadFloat32x8SlicePart(c[j:])
						av.MulAdd(archsimd.LoadFloat32x8SlicePart(bb[j:]), cv).StoreSlicePart(c[j:])
					}
				}
			}
		}
	}
}
