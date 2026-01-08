# Cấu Trúc Modules - VibeVoice Colab Demo

## 📁 Cấu Trúc Thư Mục Mới

```
demo/
├── colab.py                    # Main entry point
├── colab_modules/              # ⭐ TẤT CẢ MODULES Ở ĐÂY
│   ├── __init__.py            # Package exports
│   ├── colab_config.py        # Configuration
│   ├── colab_model.py         # Model management
│   ├── colab_voice.py         # Voice management
│   ├── colab_audio.py         # Audio processing
│   ├── colab_generator.py     # Generation logic
│   ├── colab_ui.py            # ⭐ UI Components (dễ chỉnh sửa)
│   └── colab_prompt_builder.py # Prompt Builder UI
├── colab_utils/                # Utilities
│   ├── __init__.py
│   ├── download.py
│   └── file_ops.py
└── ...
```

## 🎯 Lợi Ích

1. **Gọn gàng hơn**: Tất cả modules trong 1 thư mục
2. **Dễ quản lý**: Cấu trúc rõ ràng, dễ tìm file
3. **Dễ import**: Sử dụng `from demo.colab_modules import ...`

## 📝 Cách Import

### Trong `colab.py`:
```python
from demo.colab_modules import (
    config,
    ModelManager,
    VoiceManager,
    AudioProcessor,
    PodcastGenerator,
    create_demo_interface,
    create_prompt_builder_ui
)
```

### Trong các module khác:
```python
from .colab_config import config
from .colab_model import ModelManager
```

## ✨ Tính Năng Mới: Voice Preview

Người dùng có thể **nghe preview** của các voice trước khi chọn:

1. Chọn voice từ dropdown
2. Audio preview tự động hiển thị bên dưới
3. Click play để nghe thử
4. Chọn voice phù hợp

### Cách hoạt động:
- Mỗi speaker dropdown có một audio preview component
- Khi chọn voice, preview tự động cập nhật
- Preview hiển thị/ẩn theo số lượng speakers

---

*Modules được tổ chức gọn gàng trong `colab_modules/`* 📦

