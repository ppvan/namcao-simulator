package data

import (
	"fmt"
	"os"
	"slices"
)

type Vocab struct {
	CharToIdx map[byte]int
	IdxToChar []byte
}

func LoadText(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func BuildVocab(text string) *Vocab {
	seen := make(map[byte]struct{})
	for i := 0; i < len(text); i++ {
		seen[text[i]] = struct{}{}
	}

	chars := make([]byte, 0, len(seen))
	for c := range seen {
		chars = append(chars, c)
	}
	slices.Sort(chars)

	charToIdx := make(map[byte]int, len(chars))
	for i, c := range chars {
		charToIdx[c] = i
	}

	return &Vocab{CharToIdx: charToIdx, IdxToChar: chars}
}

func (v *Vocab) VocabSize() int {
	return len(v.IdxToChar)
}

func (v *Vocab) Encode(text string) ([]int, error) {
	ids := make([]int, len(text))
	for i := 0; i < len(text); i++ {
		id, ok := v.CharToIdx[text[i]]
		if !ok {
			return nil, fmt.Errorf("data: byte %q at offset %d is not in the vocabulary", text[i], i)
		}
		ids[i] = id
	}
	return ids, nil
}

func Encode(text string, v *Vocab) []int {
	ids := make([]int, len(text))
	for i := 0; i < len(text); i++ {
		ids[i] = v.CharToIdx[text[i]]
	}
	return ids
}

func Decode(ids []int, v *Vocab) string {
	b := make([]byte, len(ids))
	for i, id := range ids {
		b[i] = v.IdxToChar[id]
	}
	return string(b)
}

func SplitData(ids []int, trainRatio float64) (train, val []int) {
	n := int(float64(len(ids)) * trainRatio)
	return ids[:n], ids[n:]
}

func NextBatch(ids []int, pos, seqLen int) (xs, ys []int, nextPos int, wrapped bool) {
	if len(ids) < seqLen+1 {
		panic(fmt.Sprintf("data: need at least %d ids for seqLen %d, got %d", seqLen+1, seqLen, len(ids)))
	}
	if pos < 0 || pos+seqLen+1 > len(ids) {
		pos = 0
		wrapped = true
	}
	xs = ids[pos : pos+seqLen]
	ys = ids[pos+1 : pos+seqLen+1]
	return xs, ys, pos + seqLen, wrapped
}
