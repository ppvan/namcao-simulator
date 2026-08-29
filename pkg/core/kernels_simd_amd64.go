//go:build goexperiment.simd && amd64

package core

import "simd/archsimd"

var simdOK = archsimd.X86.AVX2() && archsimd.X86.FMA()

const simdLanes = 8

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

		for ; j+5 < q; j += 6 {
			b0 := b[(j+0)*n : (j+0)*n+n]
			b1 := b[(j+1)*n : (j+1)*n+n]
			b2 := b[(j+2)*n : (j+2)*n+n]
			b3 := b[(j+3)*n : (j+3)*n+n]
			b4 := b[(j+4)*n : (j+4)*n+n]
			b5 := b[(j+5)*n : (j+5)*n+n]
			var s0, s1, s2, s3, s4, s5 archsimd.Float32x8
			k := 0
			for ; k+simdLanes <= n; k += simdLanes {
				av := archsimd.LoadFloat32x8Slice(arow[k:])
				s0 = av.MulAdd(archsimd.LoadFloat32x8Slice(b0[k:]), s0)
				s1 = av.MulAdd(archsimd.LoadFloat32x8Slice(b1[k:]), s1)
				s2 = av.MulAdd(archsimd.LoadFloat32x8Slice(b2[k:]), s2)
				s3 = av.MulAdd(archsimd.LoadFloat32x8Slice(b3[k:]), s3)
				s4 = av.MulAdd(archsimd.LoadFloat32x8Slice(b4[k:]), s4)
				s5 = av.MulAdd(archsimd.LoadFloat32x8Slice(b5[k:]), s5)
			}
			if k < n {
				av := archsimd.LoadFloat32x8SlicePart(arow[k:])
				s0 = av.MulAdd(archsimd.LoadFloat32x8SlicePart(b0[k:]), s0)
				s1 = av.MulAdd(archsimd.LoadFloat32x8SlicePart(b1[k:]), s1)
				s2 = av.MulAdd(archsimd.LoadFloat32x8SlicePart(b2[k:]), s2)
				s3 = av.MulAdd(archsimd.LoadFloat32x8SlicePart(b3[k:]), s3)
				s4 = av.MulAdd(archsimd.LoadFloat32x8SlicePart(b4[k:]), s4)
				s5 = av.MulAdd(archsimd.LoadFloat32x8SlicePart(b5[k:]), s5)
			}
			crow[j+0] = hsum8(s0, &lanes)
			crow[j+1] = hsum8(s1, &lanes)
			crow[j+2] = hsum8(s2, &lanes)
			crow[j+3] = hsum8(s3, &lanes)
			crow[j+4] = hsum8(s4, &lanes)
			crow[j+5] = hsum8(s5, &lanes)
		}

		for ; j+1 < q; j += 2 {
			b0 := b[(j+0)*n : (j+0)*n+n]
			b1 := b[(j+1)*n : (j+1)*n+n]
			var s0, s1 archsimd.Float32x8
			k := 0
			for ; k+simdLanes <= n; k += simdLanes {
				av := archsimd.LoadFloat32x8Slice(arow[k:])
				s0 = av.MulAdd(archsimd.LoadFloat32x8Slice(b0[k:]), s0)
				s1 = av.MulAdd(archsimd.LoadFloat32x8Slice(b1[k:]), s1)
			}
			if k < n {
				av := archsimd.LoadFloat32x8SlicePart(arow[k:])
				s0 = av.MulAdd(archsimd.LoadFloat32x8SlicePart(b0[k:]), s0)
				s1 = av.MulAdd(archsimd.LoadFloat32x8SlicePart(b1[k:]), s1)
			}
			crow[j+0] = hsum8(s0, &lanes)
			crow[j+1] = hsum8(s1, &lanes)
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

func hsum8(v archsimd.Float32x8, lanes *[simdLanes]float32) float32 {
	v.StoreSlice(lanes[:])
	return ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
		((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
}

func tDotAccInto(a, b, c []float32, m, n, p int) {
	if !simdOK {
		tDotAccIntoScalar(a, b, c, m, n, p)
		return
	}
	for i := 0; i < n; i++ {
		crow := c[i*p : i*p+p]
		k := 0

		for ; k+5 < m; k += 6 {
			a0 := archsimd.BroadcastFloat32x8(a[(k+0)*n+i])
			a1 := archsimd.BroadcastFloat32x8(a[(k+1)*n+i])
			a2 := archsimd.BroadcastFloat32x8(a[(k+2)*n+i])
			a3 := archsimd.BroadcastFloat32x8(a[(k+3)*n+i])
			a4 := archsimd.BroadcastFloat32x8(a[(k+4)*n+i])
			a5 := archsimd.BroadcastFloat32x8(a[(k+5)*n+i])
			b0 := b[(k+0)*p : (k+0)*p+p]
			b1 := b[(k+1)*p : (k+1)*p+p]
			b2 := b[(k+2)*p : (k+2)*p+p]
			b3 := b[(k+3)*p : (k+3)*p+p]
			b4 := b[(k+4)*p : (k+4)*p+p]
			b5 := b[(k+5)*p : (k+5)*p+p]
			j := 0
			for ; j+simdLanes <= p; j += simdLanes {
				cv := archsimd.LoadFloat32x8Slice(crow[j:])
				cv = a0.MulAdd(archsimd.LoadFloat32x8Slice(b0[j:]), cv)
				cv = a1.MulAdd(archsimd.LoadFloat32x8Slice(b1[j:]), cv)
				cv = a2.MulAdd(archsimd.LoadFloat32x8Slice(b2[j:]), cv)
				cv = a3.MulAdd(archsimd.LoadFloat32x8Slice(b3[j:]), cv)
				cv = a4.MulAdd(archsimd.LoadFloat32x8Slice(b4[j:]), cv)
				cv = a5.MulAdd(archsimd.LoadFloat32x8Slice(b5[j:]), cv)
				cv.StoreSlice(crow[j:])
			}
			if j < p {
				cv := archsimd.LoadFloat32x8SlicePart(crow[j:])
				cv = a0.MulAdd(archsimd.LoadFloat32x8SlicePart(b0[j:]), cv)
				cv = a1.MulAdd(archsimd.LoadFloat32x8SlicePart(b1[j:]), cv)
				cv = a2.MulAdd(archsimd.LoadFloat32x8SlicePart(b2[j:]), cv)
				cv = a3.MulAdd(archsimd.LoadFloat32x8SlicePart(b3[j:]), cv)
				cv = a4.MulAdd(archsimd.LoadFloat32x8SlicePart(b4[j:]), cv)
				cv = a5.MulAdd(archsimd.LoadFloat32x8SlicePart(b5[j:]), cv)
				cv.StoreSlicePart(crow[j:])
			}
		}

		for ; k < m; k++ {
			aki := a[k*n+i]
			if aki == 0 {
				continue
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

				for ; k+5 < kMax; k += 6 {
					a0 := archsimd.BroadcastFloat32x8(a[aRow+k])
					a1 := archsimd.BroadcastFloat32x8(a[aRow+k+1])
					a2 := archsimd.BroadcastFloat32x8(a[aRow+k+2])
					a3 := archsimd.BroadcastFloat32x8(a[aRow+k+3])
					a4 := archsimd.BroadcastFloat32x8(a[aRow+k+4])
					a5 := archsimd.BroadcastFloat32x8(a[aRow+k+5])
					b0 := b[(k+0)*p+colStart : (k+0)*p+colEnd]
					b1 := b[(k+1)*p+colStart : (k+1)*p+colEnd]
					b2 := b[(k+2)*p+colStart : (k+2)*p+colEnd]
					b3 := b[(k+3)*p+colStart : (k+3)*p+colEnd]
					b4 := b[(k+4)*p+colStart : (k+4)*p+colEnd]
					b5 := b[(k+5)*p+colStart : (k+5)*p+colEnd]
					j := 0
					for ; j+simdLanes <= width; j += simdLanes {
						cv := archsimd.LoadFloat32x8Slice(c[j:])
						cv = a0.MulAdd(archsimd.LoadFloat32x8Slice(b0[j:]), cv)
						cv = a1.MulAdd(archsimd.LoadFloat32x8Slice(b1[j:]), cv)
						cv = a2.MulAdd(archsimd.LoadFloat32x8Slice(b2[j:]), cv)
						cv = a3.MulAdd(archsimd.LoadFloat32x8Slice(b3[j:]), cv)
						cv = a4.MulAdd(archsimd.LoadFloat32x8Slice(b4[j:]), cv)
						cv = a5.MulAdd(archsimd.LoadFloat32x8Slice(b5[j:]), cv)
						cv.StoreSlice(c[j:])
					}
					if j < width {
						cv := archsimd.LoadFloat32x8SlicePart(c[j:])
						cv = a0.MulAdd(archsimd.LoadFloat32x8SlicePart(b0[j:]), cv)
						cv = a1.MulAdd(archsimd.LoadFloat32x8SlicePart(b1[j:]), cv)
						cv = a2.MulAdd(archsimd.LoadFloat32x8SlicePart(b2[j:]), cv)
						cv = a3.MulAdd(archsimd.LoadFloat32x8SlicePart(b3[j:]), cv)
						cv = a4.MulAdd(archsimd.LoadFloat32x8SlicePart(b4[j:]), cv)
						cv = a5.MulAdd(archsimd.LoadFloat32x8SlicePart(b5[j:]), cv)
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
