package core

func dotTIntoScalar(a, b, c []float32, m, n, q int) {
	for i := 0; i < m; i++ {
		arow := a[i*n : i*n+n]
		crow := c[i*q : i*q+q]
		for j := 0; j < q; j++ {
			brow := b[j*n : j*n+n]
			var s0, s1, s2, s3 float32
			k := 0
			for ; k+3 < n; k += 4 {
				s0 += arow[k] * brow[k]
				s1 += arow[k+1] * brow[k+1]
				s2 += arow[k+2] * brow[k+2]
				s3 += arow[k+3] * brow[k+3]
			}
			for ; k < n; k++ {
				s0 += arow[k] * brow[k]
			}
			crow[j] = (s0 + s1) + (s2 + s3)
		}
	}
}

func tDotAccIntoScalar(a, b, c []float32, m, n, p int) {
	for k := 0; k < m; k++ {
		arow := a[k*n : k*n+n]
		brow := b[k*p : k*p+p]
		for i := 0; i < n; i++ {
			aki := arow[i]
			if aki == 0 {
				continue
			}
			crow := c[i*p : i*p+p]
			for j := range crow {
				crow[j] += aki * brow[j]
			}
		}
	}
}

func dotTiledScalar(a, b, result []float32, rowStart, rowEnd, n, p, colStart, colEnd int) {
	for ii := rowStart; ii < rowEnd; ii += tileM {
		iMax := min(ii+tileM, rowEnd)
		for kk := 0; kk < n; kk += tileK {
			kMax := min(kk+tileK, n)
			for i := ii; i < iMax; i++ {
				aRow := i * n
				c := result[i*p+colStart : i*p+colEnd]

				k := kk
				for ; k+3 < kMax; k += 4 {
					a0 := a[aRow+k]
					a1 := a[aRow+k+1]
					a2 := a[aRow+k+2]
					a3 := a[aRow+k+3]
					b0 := b[(k+0)*p+colStart : (k+0)*p+colEnd]
					b1 := b[(k+1)*p+colStart : (k+1)*p+colEnd]
					b2 := b[(k+2)*p+colStart : (k+2)*p+colEnd]
					b3 := b[(k+3)*p+colStart : (k+3)*p+colEnd]
					for j := range c {
						c[j] += a0*b0[j] + a1*b1[j] + a2*b2[j] + a3*b3[j]
					}
				}
				for ; k < kMax; k++ {
					aik := a[aRow+k]
					bb := b[k*p+colStart : k*p+colEnd]
					for j := range c {
						c[j] += aik * bb[j]
					}
				}
			}
		}
	}
}
