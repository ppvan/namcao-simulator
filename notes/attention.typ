== Attention

=== Vì sao Attention ra đời?

Trước đây, trong bài toán dịch máy (Machine Translation), người ta thường dùng kiến trúc Bi-LSTM, dạng Encoder - Decoder, tức là 2 LSTM nối lại với nhau.

Ví dụ, để dịch câu "hello world":

Encoder sẽ tokenize, sau đó compress input thành state vector. Bản chất của LSTM là "tóm tắt" input về một "hidden state vector".

$ ["hello", "world"] -> h_1 -> h_2 -> ... -> h_n $

Lưu ý: encoder thường là *Bi-LSTM* (2 chiều) — chạy forward và backward rồi ghép lại — vì mỗi $h_i$ chạy 1 chiều chỉ "biết" các token trước (hoặc sau) nó, ghép 2 chiều thì $h_i$ mới thực sự nắm được toàn câu. Decoder thì luôn là 1 chiều (unidirectional), vì nó autoregressive — sinh từng từ dựa trên từ trước đó, không thể nhìn thấy từ tương lai.

Hidden state cuối cùng được tổng hợp thành "context vector" và pass cho decoder để ra ngôn ngữ đích:

$ h_n -> c_v -> s_1 -> s_2 -> ... -> s_n -> ["xin", "chào", "thế", "giới"] $

Ta thấy rằng, LSTM cố nén toàn bộ thông tin thành 1 vector duy nhất trước khi dịch — seq càng dài, khả năng mất thông tin càng lớn (vì phải nén rất nhiều token vào 1 vector kích thước cố định).

Thế là người ta nghĩ ra attention — cơ chế cho phép decoder "nhìn" tất cả các hidden state của encoder, thay vì chỉ state cuối cùng:

$ {h_1, h_2, h_3, ..., h_n} $

Model sẽ tự học cách "nhìn nhận" state nào là quan trọng, từ dữ liệu.

=== Attention là gì?

Điểm quan trọng nhất cần hiểu: attention *không phải* một bước xử lý duy nhất chèn giữa encoder và decoder. Cái làm nên khác biệt là — *mỗi bước decode có một context vector riêng*, được tính động dựa trên decoder đang ở bước nào. Đây chính là điểm khác biệt cốt lõi so với seq2seq thường (chỉ có 1 context vector duy nhất cho cả câu).

Công thức (Bahdanau attention), tại mỗi bước decode $t$:

$ "score"_i &= "align"(s_(t-1), h_i) \
alpha_i &= "softmax"("score"_1, ..., "score"_n) \
c_t &= sum_(i=1)^n alpha_i h_i \
s_t &= "LSTM"_"decoder" (s_(t-1), y_(t-1), c_t) $

trong đó $s_(t-1)$ là decoder state ở bước trước, $h_i$ là từng hidden state của encoder, $alpha_i$ là trọng số attention (tổng bằng 1), và $c_t$ là context vector *riêng cho bước $t$*.

Sơ đồ đúng hơn (dạng vòng lặp, không phải pipeline thẳng):

```
{h_1, ..., h_n}  ← cố định, encoder chỉ chạy 1 lần
        ↓
   bước t=1: c_1 = attend(s_0, {h_i}) → s_1 → "xin"
   bước t=2: c_2 = attend(s_1, {h_i}) → s_2 → "chào"
   bước t=3: c_3 = attend(s_2, {h_i}) → s_3 → "thế"
   bước t=4: c_4 = attend(s_3, {h_i}) → s_4 → "giới"
```

Điểm mấu chốt: $\{h_1, ..., h_n\}$ chỉ tính *một lần* (encoder chạy 1 lần duy nhất), nhưng $alpha$ và $c_t$ được tính *lại ở mỗi bước decode* — vì mỗi từ tiếng đích cần nhìn vào phần khác nhau của câu nguồn. Ví dụ dịch "hello world" → "xin chào thế giới": lúc sinh "chào", model có thể attend mạnh vào "hello"; lúc sinh "giới", attend mạnh vào "world".

=== Từ Bahdanau attention đến Self-Attention

Đây là 2 bước tiến hoá riêng biệt, không nên gộp làm một:

+ *Bahdanau (2014)* — thêm attention vào kiến trúc LSTM encoder-decoder có sẵn: decoder attend vào encoder (gọi là *cross-attention*). LSTM vẫn còn nguyên, attention chỉ là phần bổ sung ở cuối.

+ *Vaswani et al., "Attention is All You Need" (2017)* — bỏ hẳn LSTM, dùng attention làm cơ chế chính. Không chỉ decoder attend vào encoder, mà encoder tự attend vào chính nó, decoder tự attend vào chính nó — đây là lúc *self-attention* mới thực sự đúng nghĩa (Bahdanau chỉ có cross-attention, chưa có self-attention).

Vì bỏ LSTM nghĩa là bỏ luôn cơ chế xử lý tuần tự vốn ngầm mã hoá thứ tự câu, self-attention cần *positional encoding* cộng thẳng vào embedding để bù lại. Nếu không, "hello world" và "world hello" sẽ cho ra attention score y hệt nhau — self-attention là *permutation-invariant* theo bản chất.

Do ta chỉ quan tâm đến transformer, ta sẽ đi thẳng đến self-attention (Q, K, V).