package rnn

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/ppvan/gpt/pkg/core"
	"github.com/ppvan/gpt/pkg/data"
)

const (
	ckptMagic   = "GPTCKPT\x00"
	ckptVersion = uint32(1)
)

type ckptArch struct {
	VocabSize  int `json:"vocab_size"`
	EmbedSize  int `json:"embed_size"`
	HiddenSize int `json:"hidden_size"`
}

type tensorEntry struct {
	Rows   int `json:"rows"`
	Cols   int `json:"cols"`
	Offset int `json:"offset"`
	Len    int `json:"len"` // == Rows*Cols, redundant but handy for validation
}

type adamState struct {
	M map[string]tensorEntry `json:"m"`
	V map[string]tensorEntry `json:"v"`
	T map[string]int         `json:"t"`
}

type ckptVocab struct {
	TokenToIdx map[string]int `json:"token_to_idx"`
	IdxToToken []string       `json:"idx_to_token"`
}

type ckptOptions struct {
	Lowercase bool `json:"lowercase"`
	MinFreq   int  `json:"min_freq"`
}

type ckptHeader struct {
	Arch    ckptArch               `json:"arch"`
	Vocab   *ckptVocab             `json:"vocab,omitempty"`
	Options *ckptOptions           `json:"options,omitempty"`
	Tensors map[string]tensorEntry `json:"tensors"`
	Adam    *adamState             `json:"adam,omitempty"`
}

type SaveOptions struct {
	// Vocab, if non-nil, is embedded in the checkpoint header so the file is
	// fully self-contained and can be used for inference without the original
	// text corpus.
	Vocab *data.Vocab

	// TokenizeOptions is embedded alongside Vocab so a prompt at inference
	// time can be tokenized (lowercased, etc.) exactly the way the training
	// corpus was. Only meaningful when Vocab is also set.
	TokenizeOptions data.Options

	// Optimizer, if non-nil and pointing to a *core.Adam, causes the Adam
	// moment buffers and step counters to be written alongside the weights
	// so that training can be resumed without warmup bias.
	Optimizer core.Optimizer
}

func Save(path string, m *CharRNN, opts SaveOptions) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("checkpoint save: create %q: %w", path, err)
	}
	defer f.Close()
	return write(f, m, opts)
}

type LoadResult struct {
	Model *CharRNN
	// Vocab is non-nil when the checkpoint was saved with a Vocab.
	// It is needed for inference (encoding prompts and decoding output).
	Vocab *data.Vocab
	// TokenizeOptions is populated whenever Vocab is, and should be passed
	// to data.Tokenize/Vocab.Encode when processing a prompt so it matches
	// how the training corpus was tokenized.
	TokenizeOptions data.Options
	// Adam is non-nil only when the checkpoint contained optimizer state.
	Adam *core.Adam
}

func Load(path string) (LoadResult, error) {
	f, err := os.Open(path)
	if err != nil {
		return LoadResult{}, fmt.Errorf("checkpoint load: open %q: %w", path, err)
	}
	defer f.Close()
	return read(f)
}

func write(w io.Writer, m *CharRNN, opts SaveOptions) error {
	// ── 1. collect tensors ───────────────────────────────────────────────
	params := m.Params()

	var blob []float32
	tensors := make(map[string]tensorEntry, len(params))

	for name, mat := range params {
		off := len(blob)
		raw := matWeights(mat)
		blob = append(blob, raw...)
		tensors[name] = tensorEntry{
			Rows:   mat.Rows,
			Cols:   mat.Columns,
			Offset: off,
			Len:    len(raw),
		}
	}

	// ── 2. optional Adam state ───────────────────────────────────────────
	var hdrAdam *adamState
	if adam, ok := opts.Optimizer.(*core.Adam); ok && adam != nil {
		hdrAdam = &adamState{
			M: make(map[string]tensorEntry),
			V: make(map[string]tensorEntry),
			T: adam.Steps(),
		}
		for name, mat := range adam.M() {
			off := len(blob)
			raw := matWeights(mat)
			blob = append(blob, raw...)
			hdrAdam.M[name] = tensorEntry{Rows: mat.Rows, Cols: mat.Columns, Offset: off, Len: len(raw)}
		}
		for name, mat := range adam.V() {
			off := len(blob)
			raw := matWeights(mat)
			blob = append(blob, raw...)
			hdrAdam.V[name] = tensorEntry{Rows: mat.Rows, Cols: mat.Columns, Offset: off, Len: len(raw)}
		}
	}

	// ── 3. encode header JSON ────────────────────────────────────────────
	var hdrVocab *ckptVocab
	var hdrOptions *ckptOptions
	if opts.Vocab != nil {
		hdrVocab = &ckptVocab{
			TokenToIdx: opts.Vocab.TokenToIdx,
			IdxToToken: opts.Vocab.IdxToToken,
		}
		hdrOptions = &ckptOptions{
			Lowercase: opts.TokenizeOptions.Lowercase,
			MinFreq:   opts.TokenizeOptions.MinFreq,
		}
	}

	hdr := ckptHeader{
		Arch: ckptArch{
			VocabSize:  m.VocabSize(),
			EmbedSize:  m.Embed.Weights.Columns,
			HiddenSize: m.HiddenSize(),
		},
		Vocab:   hdrVocab,
		Options: hdrOptions,
		Tensors: tensors,
		Adam:    hdrAdam,
	}
	hdrJSON, err := json.Marshal(hdr)
	if err != nil {
		return fmt.Errorf("checkpoint save: marshal header: %w", err)
	}

	// ── 4. write file ────────────────────────────────────────────────────
	// magic
	if _, err := io.WriteString(w, ckptMagic); err != nil {
		return fmt.Errorf("checkpoint save: write magic: %w", err)
	}
	// version
	if err := binary.Write(w, binary.LittleEndian, ckptVersion); err != nil {
		return fmt.Errorf("checkpoint save: write version: %w", err)
	}
	// header length
	hdrLen := uint32(len(hdrJSON))
	if err := binary.Write(w, binary.LittleEndian, hdrLen); err != nil {
		return fmt.Errorf("checkpoint save: write header length: %w", err)
	}
	// header
	if _, err := w.Write(hdrJSON); err != nil {
		return fmt.Errorf("checkpoint save: write header: %w", err)
	}
	// tensor blob
	if err := binary.Write(w, binary.LittleEndian, blob); err != nil {
		return fmt.Errorf("checkpoint save: write tensor blob: %w", err)
	}

	return nil
}

func read(r io.Reader) (LoadResult, error) {
	// magic
	magic := make([]byte, len(ckptMagic))
	if _, err := io.ReadFull(r, magic); err != nil {
		return LoadResult{}, fmt.Errorf("checkpoint load: read magic: %w", err)
	}
	if string(magic) != ckptMagic {
		return LoadResult{}, fmt.Errorf("checkpoint load: bad magic %q", magic)
	}

	// version
	var version uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return LoadResult{}, fmt.Errorf("checkpoint load: read version: %w", err)
	}
	if version != ckptVersion {
		return LoadResult{}, fmt.Errorf("checkpoint load: unsupported version %d (want %d)", version, ckptVersion)
	}

	// header length
	var hdrLen uint32
	if err := binary.Read(r, binary.LittleEndian, &hdrLen); err != nil {
		return LoadResult{}, fmt.Errorf("checkpoint load: read header length: %w", err)
	}

	// header JSON
	hdrJSON := make([]byte, hdrLen)
	if _, err := io.ReadFull(r, hdrJSON); err != nil {
		return LoadResult{}, fmt.Errorf("checkpoint load: read header: %w", err)
	}
	var hdr ckptHeader
	if err := json.Unmarshal(hdrJSON, &hdr); err != nil {
		return LoadResult{}, fmt.Errorf("checkpoint load: parse header: %w", err)
	}

	// tensor blob – read the rest of the file
	blobBytes, err := io.ReadAll(r)
	if err != nil {
		return LoadResult{}, fmt.Errorf("checkpoint load: read tensor blob: %w", err)
	}
	if len(blobBytes)%4 != 0 {
		return LoadResult{}, fmt.Errorf("checkpoint load: blob size %d is not a multiple of 4", len(blobBytes))
	}
	blob := make([]float32, len(blobBytes)/4)
	for i := range blob {
		bits := binary.LittleEndian.Uint32(blobBytes[i*4 : i*4+4])
		blob[i] = float32frombits(bits)
	}

	// ── reconstruct model ────────────────────────────────────────────────
	arch := hdr.Arch
	model := NewCharLSTM(arch.VocabSize, arch.EmbedSize, arch.HiddenSize)

	params := make(core.Params, len(hdr.Tensors))
	for name, e := range hdr.Tensors {
		if err := validateEntry(e, len(blob)); err != nil {
			return LoadResult{}, fmt.Errorf("checkpoint load: tensor %q: %w", name, err)
		}
		raw := make([]float32, e.Len)
		copy(raw, blob[e.Offset:e.Offset+e.Len])
		params[name] = core.NewMat(e.Rows, e.Cols, raw)
	}
	model.SetParams(params)

	// ── reconstruct Adam (optional) ──────────────────────────────────────
	var restoredAdam *core.Adam
	if hdr.Adam != nil {
		adam := core.NewAdam(0, 0, 0, 0) // hyperparams are stored externally; caller sets them
		mMap := make(map[string]core.Mat, len(hdr.Adam.M))
		for name, e := range hdr.Adam.M {
			if err := validateEntry(e, len(blob)); err != nil {
				return LoadResult{}, fmt.Errorf("checkpoint load: adam.m %q: %w", name, err)
			}
			raw := make([]float32, e.Len)
			copy(raw, blob[e.Offset:e.Offset+e.Len])
			mMap[name] = core.NewMat(e.Rows, e.Cols, raw)
		}
		vMap := make(map[string]core.Mat, len(hdr.Adam.V))
		for name, e := range hdr.Adam.V {
			if err := validateEntry(e, len(blob)); err != nil {
				return LoadResult{}, fmt.Errorf("checkpoint load: adam.v %q: %w", name, err)
			}
			raw := make([]float32, e.Len)
			copy(raw, blob[e.Offset:e.Offset+e.Len])
			vMap[name] = core.NewMat(e.Rows, e.Cols, raw)
		}
		adam.SetState(mMap, vMap, hdr.Adam.T)
		restoredAdam = adam
	}

	// ── restore vocab + tokenize options (optional) ──────────────────────
	var restoredVocab *data.Vocab
	var restoredOptions data.Options
	if hdr.Vocab != nil {
		cv := hdr.Vocab
		restoredVocab = &data.Vocab{
			TokenToIdx: cv.TokenToIdx,
			IdxToToken: cv.IdxToToken,
		}
	}
	if hdr.Options != nil {
		restoredOptions = data.Options{
			Lowercase: hdr.Options.Lowercase,
			MinFreq:   hdr.Options.MinFreq,
		}
	}

	return LoadResult{
		Model:           model,
		Vocab:           restoredVocab,
		TokenizeOptions: restoredOptions,
		Adam:            restoredAdam,
	}, nil
}

func matWeights(m core.Mat) []float32 {
	out := make([]float32, m.Rows*m.Columns)
	for r := 0; r < m.Rows; r++ {
		for c := 0; c < m.Columns; c++ {
			out[r*m.Columns+c] = m.Get(r, c)
		}
	}
	return out
}

func validateEntry(e tensorEntry, blobLen int) error {
	if e.Len != e.Rows*e.Cols {
		return fmt.Errorf("len %d != rows %d * cols %d", e.Len, e.Rows, e.Cols)
	}
	if e.Offset < 0 || e.Offset+e.Len > blobLen {
		return fmt.Errorf("offset %d + len %d out of blob range %d", e.Offset, e.Len, blobLen)
	}
	return nil
}

func float32frombits(b uint32) float32 { return math.Float32frombits(b) }
