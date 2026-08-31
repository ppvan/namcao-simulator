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

type Options struct {
	Lowercase bool

	MinFreq int
}

var DefaultOptions = Options{Lowercase: true, MinFreq: 2}

type Vocab struct {
	TokenToIdx map[string]int
	IdxToToken []string
}

func (v *Vocab) VocabSize() int {
	return len(v.IdxToToken)
}

func LoadText(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

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

func Normalize(text string, opts Options) string {
	if opts.Lowercase {
		text = strings.ToLower(text)
	}
	return text
}

var keepPunct = map[rune]struct{}{
	'.': {}, ',': {}, '!': {}, '?': {}, ';': {}, ':': {},
	'…': {}, '—': {}, '–': {}, '-': {},
}

var SentenceEnders = map[string]struct{}{
	".": {}, "!": {}, "?": {}, "…": {},
}

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

func BuildVocab(text string, opts Options) *Vocab {
	if opts.MinFreq < 1 {
		opts.MinFreq = 1
	}

	tokens := Tokenize(text, opts)

	freq := make(map[string]int, len(tokens))
	for _, t := range tokens {
		if isSpecialToken(t) {

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

type SeqBatcher struct {
	ids    []int
	batch  int
	seqLen int

	span   int
	starts []int
	offset int

	xbuf, ybuf   []int
	xrows, yrows [][]int
}

func NewSeqBatcher(ids []int, batch, seqLen int) (*SeqBatcher, error) {
	if batch <= 0 {
		batch = 1
	}
	if seqLen <= 0 {
		return nil, fmt.Errorf("data: seqLen must be positive, got %d", seqLen)
	}

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

func (b *SeqBatcher) WindowsPerEpoch() int { return b.span / b.seqLen }

func (b *SeqBatcher) Reset() { b.offset = 0 }

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
