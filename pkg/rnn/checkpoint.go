// Package rnn – checkpoint.go
//
// Binary checkpoint format (project-specific, not an open standard).
//
// Layout
// ──────
//
//	Offset  Size  Field
//	     0     8  Magic bytes  "GPTCKPT\x00"
//	     8     4  Format version (uint32 LE) – currently 1
//	    12     4  Header length H (uint32 LE)
//	    16     H  Header JSON (UTF-8)
//	  16+H     *  Tensor blob – packed float32 values (little-endian)
//
// Header JSON schema
// ──────────────────
//
//	{
//	  "arch": { "vocab_size": int, "embed_size": int, "hidden_size": int },
//	  "vocab": {
//	    "char_to_idx": { "<char>": int, … },
//	    "idx_to_char": [ int, … ]   // byte values as ints
//	  },
//	  "tensors": {
//	    "<param_name>": { "rows": int, "cols": int, "offset": int, "len": int }
//	  },
//	  "adam": { "m": {…}, "v": {…}, "t": {…} }   // optional
//	}
//
// Rationale
// ─────────
//  • One file, no side-cars. The vocab is embedded in the JSON header so
//    the checkpoint is fully self-contained: load it anywhere and generate.
//  • Human-readable header; hexdump the first ~2 KB and you see the full
//    architecture, vocabulary, and tensor inventory.
//  • The tensor blob is a single contiguous []float32 written with
//    encoding/binary – no reflection, no unsafe, no cgo.
//  • Adam state is optional: omit for a weights-only checkpoint,
//    include to resume training exactly.

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

// ── constants ────────────────────────────────────────────────────────────────

const (
	ckptMagic   = "GPTCKPT\x00"
	ckptVersion = uint32(1)
)

// ── header types (serialised as JSON) ────────────────────────────────────────

type ckptArch struct {
	VocabSize  int `json:"vocab_size"`
	EmbedSize  int `json:"embed_size"`
	HiddenSize int `json:"hidden_size"`
}

// tensorEntry describes one named tensor inside the flat blob.
// Offset and Len are in float32 units (multiply by 4 for byte offsets).
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

// ckptVocab is the wire representation of data.Vocab embedded in the header.
// Using []int for IdxToChar avoids JSON encoding byte arrays as base64.
type ckptVocab struct {
	CharToIdx map[string]int `json:"char_to_idx"`
	IdxToChar []int          `json:"idx_to_char"`
}

type ckptHeader struct {
	Arch    ckptArch               `json:"arch"`
	Vocab   *ckptVocab             `json:"vocab,omitempty"`
	Tensors map[string]tensorEntry `json:"tensors"`
	Adam    *adamState             `json:"adam,omitempty"`
}

// ── SaveOptions ───────────────────────────────────────────────────────────────

// SaveOptions controls what gets written into the checkpoint.
type SaveOptions struct {
	// Vocab, if non-nil, is embedded in the checkpoint header so the file is
	// fully self-contained and can be used for inference without the original
	// text corpus.
	Vocab *data.Vocab

	// Optimizer, if non-nil and pointing to a *core.Adam, causes the Adam
	// moment buffers and step counters to be written alongside the weights
	// so that training can be resumed without warmup bias.
	Optimizer core.Optimizer
}

// ── Save ─────────────────────────────────────────────────────────────────────

// Save writes a checkpoint of m to path.
// If opts.Optimizer is a *core.Adam its internal state is included so
// that training can be resumed via Load + LoadAdam.
func Save(path string, m *CharRNN, opts SaveOptions) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("checkpoint save: create %q: %w", path, err)
	}
	defer f.Close()
	return write(f, m, opts)
}

// ── Load ─────────────────────────────────────────────────────────────────────

// LoadResult bundles everything read back from a checkpoint.
type LoadResult struct {
	Model *CharRNN
	// Vocab is non-nil when the checkpoint was saved with a Vocab.
	// It is needed for inference (encoding prompts and decoding output).
	Vocab *data.Vocab
	// Adam is non-nil only when the checkpoint contained optimizer state.
	Adam *core.Adam
}

// Load reads a checkpoint from path and reconstructs the CharRNN.
// If the file contains Adam state, LoadResult.Adam is populated and can
// be passed directly to CharRNN.Train.
func Load(path string) (LoadResult, error) {
	f, err := os.Open(path)
	if err != nil {
		return LoadResult{}, fmt.Errorf("checkpoint load: open %q: %w", path, err)
	}
	defer f.Close()
	return read(f)
}

// ── internal write ────────────────────────────────────────────────────────────

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
	if opts.Vocab != nil {
		cv := &ckptVocab{
			CharToIdx: make(map[string]int, len(opts.Vocab.CharToIdx)),
			IdxToChar: make([]int, len(opts.Vocab.IdxToChar)),
		}
		for b, idx := range opts.Vocab.CharToIdx {
			cv.CharToIdx[string([]byte{b})] = idx
		}
		for i, b := range opts.Vocab.IdxToChar {
			cv.IdxToChar[i] = int(b)
		}
		hdrVocab = cv
	}

	hdr := ckptHeader{
		Arch: ckptArch{
			VocabSize:  m.VocabSize(),
			EmbedSize:  m.Embed.Weights.Columns,
			HiddenSize: m.HiddenSize(),
		},
		Vocab:   hdrVocab,
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

// ── internal read ─────────────────────────────────────────────────────────────

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

	// ── restore vocab (optional) ─────────────────────────────────────────
	var restoredVocab *data.Vocab
	if hdr.Vocab != nil {
		cv := hdr.Vocab
		charToIdx := make(map[byte]int, len(cv.CharToIdx))
		for s, idx := range cv.CharToIdx {
			if len(s) != 1 {
				return LoadResult{}, fmt.Errorf("checkpoint load: vocab key %q is not a single byte", s)
			}
			charToIdx[s[0]] = idx
		}
		idxToChar := make([]byte, len(cv.IdxToChar))
		for i, v := range cv.IdxToChar {
			idxToChar[i] = byte(v)
		}
		restoredVocab = &data.Vocab{CharToIdx: charToIdx, IdxToChar: idxToChar}
	}

	return LoadResult{Model: model, Vocab: restoredVocab, Adam: restoredAdam}, nil
}

// ── helpers ───────────────────────────────────────────────────────────────────

// matWeights returns a copy of the underlying float32 slice.
// Mat does not export its weights field, so we use Get row-by-row.
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

// float32frombits wraps math.Float32frombits for local readability.
func float32frombits(b uint32) float32 { return math.Float32frombits(b) }
