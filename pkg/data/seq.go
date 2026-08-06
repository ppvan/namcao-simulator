package data

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"
)

// Special tokens, mirroring txt.Corpus's reserved ids so both packages agree
// on layout if you ever share a vocab format between them. STORY marks the
// start of a new document (one per file loaded via LoadCorpus) so the model
// can learn story boundaries instead of seeing one endless stream of text.
const (
	PAD   = "<PAD>"
	UNK   = "<UNK>"
	BOS   = "<BOS>"
	EOS   = "<EOS>"
	STORY = "<STORY>"

	PAD_ID   = 0
	UNK_ID   = 2
	BOS_ID   = 1
	EOS_ID   = 3
	STORY_ID = 4
)

// specialTokens are literal markers that pass through tokenization untouched
// — never lowercased, never punctuation-trimmed, never counted as ordinary
// vocabulary words subject to MinFreq pruning.
var specialTokens = map[string]struct{}{
	PAD:   {},
	UNK:   {},
	BOS:   {},
	EOS:   {},
	STORY: {},
}

func isSpecialToken(s string) bool {
	_, ok := specialTokens[s]
	return ok
}

// Options controls text normalization and vocab pruning for word tokenization.
type Options struct {
	Lowercase bool
	// MinFreq drops tokens occurring fewer than MinFreq times from the
	// vocabulary; they fall back to UNK at encode time.
	MinFreq int
}

// DefaultOptions is a reasonable starting point: lowercase everything, and
// drop hapax legomena into UNK (same MinFreq txt.Corpus.BuildVocab uses).
var DefaultOptions = Options{Lowercase: true, MinFreq: 2}

type Vocab struct {
	TokenToIdx map[string]int
	IdxToToken []string
}

func (v *Vocab) VocabSize() int {
	return len(v.IdxToToken)
}

// LoadText reads a single file's contents.
func LoadText(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// LoadCorpus walks dir recursively and concatenates every .txt file it finds
// (separated by a blank line), the same file discovery txt.LoadCorpus uses.
// Each file is treated as one story: its content is prefixed with the
// STORY marker so the tokenized stream carries explicit document
// boundaries. It returns one big string rather than per-sentence records
// since this package trains on a flat token stream, not sentence pairs.
func LoadCorpus(dir string) (string, error) {
	var sb strings.Builder
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return fmt.Errorf("walk %q: %w", path, err)
		}
		if d.IsDir() || !strings.HasSuffix(d.Name(), ".txt") {
			return nil
		}

		content, err := loadFile(path)
		if err != nil {
			return fmt.Errorf("load %q: %w", path, err)
		}
		if sb.Len() > 0 {
			sb.WriteString("\n\n")
		}
		sb.WriteString(STORY)
		sb.WriteString("\n\n")
		sb.WriteString(content)
		return nil
	})
	if err != nil {
		return "", err
	}
	return sb.String(), nil
}

func loadFile(path string) (string, error) {
	handle, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer handle.Close()

	b, err := io.ReadAll(handle)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// Normalize applies the options' text normalization (currently just
// lowercasing) to a standalone string. Tokenize does this per-token instead
// so it can skip special tokens like <STORY>, but Normalize is kept as a
// standalone helper for anywhere else you want the same rule applied.
func Normalize(text string, opts Options) string {
	if opts.Lowercase {
		text = strings.ToLower(text)
	}
	return text
}

// keepPunct is the punctuation emitted as standalone vocabulary tokens
// instead of being discarded. Sentence enders (. ! ? …) give the model and
// the sampler explicit sentence boundaries; commas/colons/semicolons give it
// clause rhythm; dashes cover Nam Cao's dialogue lines ("- Cậu Vàng đi đời
// rồi, ông giáo ạ!"). Quotes and brackets are still dropped: they come in
// pairs the model can't reliably balance, so keeping them just produces
// dangling quote marks in generated text.
var keepPunct = map[rune]struct{}{
	'.': {}, ',': {}, '!': {}, '?': {}, ';': {}, ':': {},
	'…': {}, '—': {}, '–': {}, '-': {},
}

// SentenceEnders are the tokens that terminate a sentence in generated
// output. Exported so generation code can count sentences while sampling.
var SentenceEnders = map[string]struct{}{
	".": {}, "!": {}, "?": {}, "…": {},
}

// Tokenize splits text into whitespace-separated word tokens, normalizing
// (lowercasing) each — except for special tokens like <STORY>, which pass
// through exactly as written. Leading/trailing punctuation is split off into
// standalone tokens (for runes in keepPunct) rather than discarded, so the
// model can learn — and emit — sentence structure. Interior punctuation
// (anh-hùng) stays inside its word.
func Tokenize(text string, opts Options) []string {
	raw := strings.Fields(text)
	tokens := make([]string, 0, len(raw)*2)
	for _, word := range raw {
		if isSpecialToken(word) {
			tokens = append(tokens, word)
			continue
		}
		if opts.Lowercase {
			word = strings.ToLower(word)
		}
		tokens = appendWordTokens(tokens, word)
	}
	return tokens
}

func isPunctOrSym(r rune) bool {
	return unicode.IsPunct(r) || unicode.IsSymbol(r)
}

// appendWordTokens splits word into up to three parts — leading punctuation,
// the core word, trailing punctuation — and appends each as tokens. Edge
// punctuation in keepPunct becomes its own token; other edge symbols are
// dropped as before.
func appendWordTokens(tokens []string, word string) []string {
	runes := []rune(word)
	i, j := 0, len(runes)
	for i < j && isPunctOrSym(runes[i]) {
		i++
	}
	for j > i && isPunctOrSym(runes[j-1]) {
		j--
	}
	tokens = appendPunctTokens(tokens, runes[:i])
	if i < j {
		tokens = append(tokens, string(runes[i:j]))
	}
	return appendPunctTokens(tokens, runes[j:])
}

// appendPunctTokens emits one token per keepPunct rune, collapsing any run
// of two or more '.' into a single '…' token (Nam Cao's "..." is an
// ellipsis, not three sentence breaks).
func appendPunctTokens(tokens []string, runes []rune) []string {
	dots := 0
	for _, r := range runes {
		if r == '.' {
			dots++
			continue
		}
		tokens = flushDots(tokens, dots)
		dots = 0
		if _, ok := keepPunct[r]; ok {
			tokens = append(tokens, string(r))
		}
	}
	return flushDots(tokens, dots)
}

func flushDots(tokens []string, dots int) []string {
	switch {
	case dots == 1:
		return append(tokens, ".")
	case dots >= 2:
		return append(tokens, "…")
	}
	return tokens
}

// BuildVocab tokenizes text per opts and builds a Vocab, reserving
// PAD/UNK/BOS/EOS/STORY at ids 0-4 the same way txt.Corpus.BuildVocab
// reserves its special tokens. Tokens occurring fewer than opts.MinFreq
// times are excluded from the vocab and fall back to UNK at encode time.
func BuildVocab(text string, opts Options) *Vocab {
	if opts.MinFreq < 1 {
		opts.MinFreq = 1
	}

	tokens := Tokenize(text, opts)

	freq := make(map[string]int, len(tokens))
	for _, t := range tokens {
		if isSpecialToken(t) {
			// Already reserved below; don't let it compete for a second id
			// or get pruned by MinFreq like an ordinary word.
			continue
		}
		freq[t]++
	}

	qualified := make([]string, 0, len(freq))
	for t, count := range freq {
		if count >= opts.MinFreq {
			qualified = append(qualified, t)
		}
	}
	sort.Strings(qualified)

	tokenToIdx := map[string]int{
		PAD:   PAD_ID,
		UNK:   UNK_ID,
		BOS:   BOS_ID,
		EOS:   EOS_ID,
		STORY: STORY_ID,
	}
	idxToToken := make([]string, 5, len(qualified)+5)
	idxToToken[PAD_ID] = PAD
	idxToToken[UNK_ID] = UNK
	idxToToken[BOS_ID] = BOS
	idxToToken[EOS_ID] = EOS
	idxToToken[STORY_ID] = STORY

	nextID := 5
	for _, t := range qualified {
		tokenToIdx[t] = nextID
		idxToToken = append(idxToToken, t)
		nextID++
	}

	return &Vocab{TokenToIdx: tokenToIdx, IdxToToken: idxToToken}
}

// Encode tokenizes text per opts and converts it to vocabulary ids, wrapped
// in BOS/EOS. Tokens absent from the vocab (below MinFreq, or unseen at
// train time) map to UNK instead of erroring, so encoding never fails on
// new input.
func (v *Vocab) Encode(text string, opts Options) []int {
	tokens := Tokenize(text, opts)
	ids := make([]int, 0, len(tokens)+2)
	ids = append(ids, BOS_ID)
	for _, t := range tokens {
		if id, ok := v.TokenToIdx[t]; ok {
			ids = append(ids, id)
		} else {
			ids = append(ids, UNK_ID)
		}
	}
	ids = append(ids, EOS_ID)
	return ids
}

// Decode converts ids back into text, dropping PAD/BOS/EOS markers and
// joining the remaining word tokens with spaces. STORY is kept in the
// output (rather than stripped like the other special tokens) since it's a
// meaningful boundary marker for generated multi-story text, not just
// sequence padding.
// noSpaceBefore lists tokens that attach to the preceding word ("rồi," not
// "rồi ,") when decoding.
var noSpaceBefore = map[string]struct{}{
	".": {}, ",": {}, "!": {}, "?": {}, ";": {}, ":": {}, "…": {},
}

func (v *Vocab) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		tok := UNK
		if id >= 0 && id < len(v.IdxToToken) {
			tok = v.IdxToToken[id]
		}
		if tok == PAD || tok == BOS || tok == EOS {
			continue
		}
		if _, tight := noSpaceBefore[tok]; !tight && sb.Len() > 0 {
			sb.WriteByte(' ')
		}
		sb.WriteString(tok)
	}
	return sb.String()
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
//
// Note: ids here are vocabulary ids (one per rune), already produced by
// Encode/(*Vocab).Encode, so this type is Unicode-agnostic — it never touches
// raw bytes or runes itself.
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
