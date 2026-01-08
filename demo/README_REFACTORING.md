# Tài Liệu Refactoring - VibeVoice Colab Demo

## 📁 Cấu Trúc File Mới

```
demo/
├── colab.py                    # Main entry point (đã refactor)
├── colab_config.py             # ⭐ TẤT CẢ SIÊU THAM SỐ Ở ĐÂY
├── colab_model.py              # Model loading và management
├── colab_voice.py              # Voice management
├── colab_audio.py              # Audio processing
├── colab_generator.py          # Generation logic
├── colab_ui.py                 # ⭐ GIAO DIỆN CHÍNH - CHỈNH SỬA Ở ĐÂY
├── colab_prompt_builder.py     # ⭐ PROMPT BUILDER UI - CHỈNH SỬA Ở ĐÂY
├── colab_utils/                # Utilities
│   ├── __init__.py
│   ├── download.py             # Download utilities
│   └── file_ops.py             # File operations
├── UI_CUSTOMIZATION_GUIDE.md  # ⭐ HƯỚNG DẪN CHỈNH SỬA GIAO DIỆN
└── README_REFACTORING.md       # File này
```

## 🎯 Mục Đích Refactoring

1. **Tách file lớn thành modules nhỏ**: Dễ đọc, dễ bảo trì
2. **Tách siêu tham số vào config**: Dễ điều chỉnh không cần sửa code
3. **Tách UI components**: Dễ chỉnh sửa giao diện

## 📝 Các Module Chính

### 1. `colab_config.py` - Configuration
**Mục đích**: Chứa TẤT CẢ các siêu tham số có thể điều chỉnh

**Các config classes**:
- `ModelConfig`: Cấu hình model (inference steps, CFG scale, etc.)
- `AudioConfig`: Cấu hình audio (sample rate, silence trimming, etc.)
- `UIConfig`: Cấu hình giao diện (text, labels, CSS, theme)
- `PromptBuilderConfig`: Cấu hình prompt builder
- `FileConfig`: Cấu hình file paths

**Cách sử dụng**:
```python
from demo.colab_modules import config

# Đọc config
cfg_scale = config.model.default_cfg_scale
sample_rate = config.audio.sample_rate
header_title = config.ui.header_title

# Sửa config (trong code hoặc file)
config.model.default_cfg_scale = 1.5
```

### 2. `colab_ui.py` - UI Components
**Mục đích**: Tạo giao diện Gradio, DỄ CHỈNH SỬA

**Các hàm chính**:
- `create_header_html()`: Tạo header HTML
- `create_settings_column()`: Tạo cột settings
- `create_generation_column()`: Tạo cột generation
- `create_usage_tips_section()`: Tạo section tips
- `create_demo_interface()`: Hàm chính tạo toàn bộ UI

**Chỉnh sửa giao diện**: Xem `UI_CUSTOMIZATION_GUIDE.md`

### 3. `colab_prompt_builder.py` - Prompt Builder UI
**Mục đích**: Tạo giao diện Prompt Builder

**Các hàm chính**:
- `build_conversation_prompt()`: Tạo prompt format
- `create_prompt_builder_ui()`: Tạo UI cho prompt builder

### 4. `colab_model.py` - Model Management
**Mục đích**: Quản lý model loading và configuration

**Class**: `ModelManager`

### 5. `colab_voice.py` - Voice Management
**Mục đích**: Quản lý voice presets và speaker selection

**Class**: `VoiceManager`

### 6. `colab_audio.py` - Audio Processing
**Mục đích**: Xử lý audio (read, trim silence, save)

**Class**: `AudioProcessor`

### 7. `colab_generator.py` - Generation Logic
**Mục đích**: Logic chính để generate podcast

**Class**: `PodcastGenerator`

### 8. `colab_utils/` - Utilities
**Mục đích**: Các hàm utility (download, file ops)

## 🔧 Cách Sử Dụng

### Chạy Demo:
```bash
python demo/colab.py --model_path microsoft/VibeVoice-1.5B --share
```

### Chỉnh sửa siêu tham số:
1. Mở `demo/colab_config.py`
2. Sửa các giá trị trong các `@dataclass`
3. Lưu và chạy lại

### Chỉnh sửa giao diện:
1. Xem `demo/UI_CUSTOMIZATION_GUIDE.md`
2. Sửa trong `demo/colab_config.py` (text, labels)
3. Hoặc sửa trong `demo/colab_ui.py` (layout, components)

## 📊 So Sánh Trước/Sau

### Trước:
- 1 file lớn 533 dòng
- Siêu tham số hardcode trong code
- Khó tìm nơi chỉnh sửa giao diện

### Sau:
- 8+ files nhỏ, mỗi file có mục đích rõ ràng
- Siêu tham số tập trung trong config
- Dễ tìm và chỉnh sửa giao diện

## ✅ Lợi Ích

1. **Dễ bảo trì**: Mỗi module có trách nhiệm riêng
2. **Dễ mở rộng**: Thêm tính năng mới không ảnh hưởng code cũ
3. **Dễ chỉnh sửa**: Config và UI tách biệt
4. **Dễ test**: Mỗi module có thể test độc lập
5. **Dễ đọc**: Code ngắn gọn, có comment rõ ràng

## 🚀 Migration Guide

Nếu bạn đã có code cũ và muốn migrate:

1. **Giữ nguyên file cũ**: File `colab.py` cũ vẫn hoạt động
2. **Sử dụng modules mới**: Import từ các module mới
3. **Chuyển siêu tham số**: Di chuyển hardcoded values vào config

## 📚 Tài Liệu Tham Khảo

- `UI_CUSTOMIZATION_GUIDE.md`: Hướng dẫn chi tiết chỉnh sửa giao diện
- `colab_config.py`: Xem tất cả config options
- Gradio Docs: https://gradio.app/docs/

---

*Refactored for better maintainability and customization* 🎨

