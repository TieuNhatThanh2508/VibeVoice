# VibeVoice Configuration Guide

File `config.py` chứa tất cả các hyperparameter và cài đặt cho VibeVoice demos. Thay đổi các giá trị trong file này để điều chỉnh hành vi mặc định của tất cả các script demo.

## Cấu trúc Config

### 1. MODEL_CONFIG
Cấu hình cho model loading:
- `model_path`: Đường dẫn model (HuggingFace ID hoặc local path)
- `device`: Device cho inference ("cuda" hoặc "cpu")
- `torch_dtype`: Kiểu dữ liệu ("bfloat16", "float16", "float32")
- `algorithm_type`: Thuật toán diffusion ("sde-dpmsolver++")
- `beta_schedule`: Lịch trình beta ("squaredcos_cap_v2")

### 2. GENERATION_CONFIG
Cấu hình cho quá trình generation:
- `ddpm_inference_steps`: Số bước diffusion (mặc định: 10)
- `cfg_scale`: CFG scale (mặc định: 1.3, range: 1.0-2.0)
- `speech_rate`: Tốc độ đọc (mặc định: 1.0, range: 0.5-2.0)
- `do_sample`: Có sampling hay không (mặc định: False)

### 3. AUDIO_CONFIG
Cấu hình cho audio:
- `sample_rate`: Tần số lấy mẫu (mặc định: 24000)
- `normalize_audio`: Có normalize audio không
- `enable_speech_rate`: Bật/tắt tính năng điều chỉnh tốc độ đọc

### 4. GRADIO_CONFIG
Cấu hình cho Gradio demo:
- `port`: Port cho server (mặc định: 7860)
- `share`: Có share public không
- `num_speakers`: Số speakers mặc định (1-4)
- `default_speakers`: Danh sách speakers mặc định
- `streaming_enabled`: Bật streaming mode

### 5. INFERENCE_CONFIG
Cấu hình cho inference script:
- `default_txt_path`: Đường dẫn file text mặc định
- `default_output_dir`: Thư mục output mặc định
- `default_speaker_names`: Tên speakers mặc định

### 6. ADVANCED_CONFIG
Cấu hình nâng cao:
- `seed`: Seed cho reproducibility (mặc định: 42)
- `log_level`: Mức độ logging
- `enable_flash_attention`: Bật flash attention

## Cách sử dụng

### 1. Thay đổi config mặc định

Chỉnh sửa file `demo/config.py`:

```python
# Ví dụ: Thay đổi CFG scale mặc định
GENERATION_CONFIG["cfg_scale"] = 1.5

# Ví dụ: Thay đổi số inference steps
GENERATION_CONFIG["ddpm_inference_steps"] = 20

# Ví dụ: Thay đổi tốc độ đọc mặc định
GENERATION_CONFIG["speech_rate"] = 1.2
```

### 2. Override bằng command line arguments

Bạn vẫn có thể override config bằng command line arguments:

```bash
# Override CFG scale
python demo/inference_from_file.py --cfg_scale 1.5

# Override model path
python demo/gradio_demo.py --model_path microsoft/VibeVoice-7B
```

### 3. Sử dụng environment variables

Một số giá trị có thể được set qua environment variables:

```bash
# Set model path
export VIBEVOICE_MODEL_PATH="microsoft/VibeVoice-1.5B"

# Set device
export VIBEVOICE_DEVICE="cuda"
```

## Validation

Config sẽ được tự động validate khi chạy. Nếu có giá trị không hợp lệ, bạn sẽ nhận được cảnh báo nhưng script vẫn tiếp tục chạy với giá trị hiện tại.

## Best Practices

1. **Backup config gốc**: Trước khi thay đổi, hãy backup file `config.py`
2. **Test từng thay đổi**: Thay đổi một tham số tại một thời điểm để dễ debug
3. **Document custom configs**: Nếu bạn tạo config riêng, hãy ghi chú lại lý do
4. **Version control**: Commit config changes vào git để track lịch sử

## Ví dụ Config Tùy chỉnh

### Config cho chất lượng cao:
```python
GENERATION_CONFIG["ddpm_inference_steps"] = 20
GENERATION_CONFIG["cfg_scale"] = 1.5
```

### Config cho tốc độ nhanh:
```python
GENERATION_CONFIG["ddpm_inference_steps"] = 5
GENERATION_CONFIG["cfg_scale"] = 1.2
```

### Config cho tốc độ đọc chậm:
```python
GENERATION_CONFIG["speech_rate"] = 0.8
```

## Troubleshooting

**Q: Config không được áp dụng?**
A: Kiểm tra xem bạn có đang override bằng command line arguments không. Command line arguments có priority cao hơn config.

**Q: Lỗi validation?**
A: Kiểm tra các giá trị trong config có nằm trong phạm vi cho phép không (ví dụ: cfg_scale phải từ 1.0-2.0).

**Q: Muốn tắt một tính năng?**
A: Set giá trị tương ứng trong config (ví dụ: `AUDIO_CONFIG["enable_speech_rate"] = False`).

