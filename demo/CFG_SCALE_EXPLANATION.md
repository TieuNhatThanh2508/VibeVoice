# CFG Scale - Giải Thích

## CFG Scale là gì?

**CFG Scale** (Classifier-Free Guidance Scale) là một tham số quan trọng trong quá trình generation của VibeVoice, giúp điều khiển mức độ "tuân thủ" của model với text input.

## Cách hoạt động

CFG Scale hoạt động dựa trên nguyên lý **Classifier-Free Guidance**:

1. **Conditional generation**: Model generate audio dựa trên text input (có điều kiện)
2. **Unconditional generation**: Model generate audio không có text input (không điều kiện)
3. **CFG Scale**: Kết hợp 2 kết quả trên với công thức:
   ```
   final_output = unconditional + cfg_scale * (conditional - unconditional)
   ```

## Giá trị CFG Scale

### CFG Scale = 1.0
- **Ý nghĩa**: Không có guidance, chỉ dùng conditional output
- **Kết quả**: Audio tự nhiên hơn, nhưng có thể ít tuân thủ text hơn

### CFG Scale = 1.3 (Mặc định)
- **Ý nghĩa**: Guidance vừa phải
- **Kết quả**: Cân bằng giữa chất lượng và tuân thủ text
- **Khuyến nghị**: Giá trị tốt cho hầu hết trường hợp

### CFG Scale > 1.3 (1.5 - 2.0)
- **Ý nghĩa**: Guidance mạnh
- **Kết quả**: 
  - ✅ Tuân thủ text tốt hơn
  - ✅ Rõ ràng hơn
  - ⚠️ Có thể hơi "cứng" hoặc mất tự nhiên
  - ⚠️ Có thể có artifacts

### CFG Scale < 1.3 (1.0 - 1.2)
- **Ý nghĩa**: Guidance yếu
- **Kết quả**:
  - ✅ Tự nhiên hơn
  - ✅ Mượt mà hơn
  - ⚠️ Có thể không tuân thủ text chính xác
  - ⚠️ Có thể bỏ sót một số từ

## Khi nào nên điều chỉnh?

### Tăng CFG Scale (1.5 - 2.0) khi:
- Text phức tạp, cần tuân thủ chính xác
- Cần rõ ràng, dễ nghe
- Script có nhiều thuật ngữ kỹ thuật
- Cần nhấn mạnh một số từ/câu

### Giảm CFG Scale (1.0 - 1.2) khi:
- Cần giọng nói tự nhiên, mượt mà
- Script đơn giản, hội thoại thông thường
- Muốn có cảm xúc tự nhiên hơn
- Audio có artifacts khi dùng CFG cao

## Trong VibeVoice

Trong code VibeVoice, CFG Scale được áp dụng trong diffusion process:

```python
# Trong modeling_vibevoice_inference.py
half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
```

Điều này có nghĩa:
- `uncond_eps`: Prediction không có text (unconditional)
- `cond_eps`: Prediction có text (conditional)
- `cfg_scale`: Độ mạnh của guidance

## Khuyến nghị

- **Mặc định**: 1.3 - Cân bằng tốt
- **Podcast thông thường**: 1.2 - 1.4
- **Script phức tạp**: 1.4 - 1.6
- **Cần tự nhiên**: 1.1 - 1.3
- **Cần chính xác**: 1.5 - 1.8

## Lưu ý

- CFG Scale quá cao (>2.0) có thể gây artifacts
- CFG Scale quá thấp (<1.0) có thể làm giảm chất lượng
- Nên test với các giá trị khác nhau để tìm giá trị phù hợp với use case của bạn

---

*CFG Scale là một công cụ mạnh để điều khiển chất lượng và style của audio generation!*

