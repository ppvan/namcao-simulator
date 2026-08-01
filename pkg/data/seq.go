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

func SplitData(ids []int, trainRatio float32) (train, val []int) {
	n := int(float32(len(ids)) * trainRatio)
	return ids[:n], ids[n:]
}

// ── batched streaming ─────────────────────────────────────────────────────────

// SeqBatcher yields time-major minibatches of B parallel character streams.
//
// The corpus is cut into B contiguous chunks and each chunk gets its own
// cursor. Every cursor advances by seqLen per call, so a stream resumes exactly
// where it stopped on the previous step — the stateful-BPTT property the
// single-stream loop had, now B times wider:
//
//	ids  ┌───── stream 0 ─────┬───── stream 1 ─────┬───── stream 2 ─────┐
//	     │ ....▓▓▓▓▓.........  │ ....▓▓▓▓▓.........  │ ....▓▓▓▓▓......... │
//	          ^offset               ^offset               ^offset
//
// Chunking (rather than sliding B adjacent windows) is what makes the streams
// decorrelated: at any step the B rows come from B distant parts of the text,
// so the gradient is an average over genuinely different contexts.
type SeqBatcher struct {
	ids    []int
	batch  int
	seqLen int

	span   int   // ids owned by each stream, excluding the +1 target lookahead
	starts []int // first index of each stream
	offset int   // shared read offset inside every stream

	// Scratch reused across Next calls; see the note on Next.
	xbuf, ybuf   []int
	xrows, yrows [][]int
}

// NewSeqBatcher splits ids into batch parallel streams of seqLen-long windows.
func NewSeqBatcher(ids []int, batch, seqLen int) (*SeqBatcher, error) {
	if batch <= 0 {
		batch = 1
	}
	if seqLen <= 0 {
		return nil, fmt.Errorf("data: seqLen must be positive, got %d", seqLen)
	}

	// Every stream needs span ids plus one lookahead for the final target.
	span := (len(ids) - 1) / batch
	if span < seqLen {
		return nil, fmt.Errorf("data: %d ids cannot feed batch %d at seqLen %d (need at least %d)",
			len(ids), batch, seqLen, batch*seqLen+1)
	}

	b := &SeqBatcher{
		ids:    ids,
		batch:  batch,
		seqLen: seqLen,
		span:   span,
		starts: make([]int, batch),
		xbuf:   make([]int, seqLen*batch),
		ybuf:   make([]int, seqLen*batch),
		xrows:  make([][]int, seqLen),
		yrows:  make([][]int, seqLen),
	}
	for i := range b.starts {
		b.starts[i] = i * span
	}
	for t := 0; t < seqLen; t++ {
		b.xrows[t] = b.xbuf[t*batch : (t+1)*batch]
		b.yrows[t] = b.ybuf[t*batch : (t+1)*batch]
	}
	return b, nil
}

func (b *SeqBatcher) Batch() int  { return b.batch }
func (b *SeqBatcher) SeqLen() int { return b.seqLen }

// WindowsPerEpoch is how many Next calls consume the corpus once.
func (b *SeqBatcher) WindowsPerEpoch() int { return b.span / b.seqLen }

// Reset rewinds every stream to the start of its chunk.
func (b *SeqBatcher) Reset() { b.offset = 0 }

// Next returns one minibatch in time-major layout: len(xs) == seqLen and each
// xs[t] holds one character per stream — exactly the [][]int that
// (*CharRNN).ForwardSeq consumes. ys[t] is the same window shifted by one.
//
// wrapped reports that the cursors were rewound before filling this batch, so
// the hidden state carried in from the previous step is stale and the caller
// should reinitialise it.
//
// The returned slices alias scratch owned by the batcher and are overwritten by
// the following call; consume or copy them within the step. The training loop
// finishes forward and backward before asking for the next batch, so this is
// safe there even though embeddingCache retains the id slices.
func (b *SeqBatcher) Next() (xs, ys [][]int, wrapped bool) {
	if b.offset+b.seqLen > b.span {
		b.offset = 0
		wrapped = true
	}

	for t := 0; t < b.seqLen; t++ {
		xt, yt := b.xrows[t], b.yrows[t]
		for i, start := range b.starts {
			p := start + b.offset + t
			xt[i] = b.ids[p]
			yt[i] = b.ids[p+1]
		}
	}

	b.offset += b.seqLen
	return b.xrows, b.yrows, wrapped
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
