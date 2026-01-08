# Hướng Dẫn Chỉnh Sửa Giao Diện VibeVoice Colab Demo

## 📍 Nơi Chỉnh Sửa Giao Diện

### 1. **File Config - `demo/colab_config.py`** ⭐ (Dễ nhất)

Đây là nơi **DỄ NHẤT** để chỉnh sửa các thông số giao diện mà không cần hiểu code:

#### Chỉnh sửa Text và Labels:
```python
@dataclass
class UIConfig:
    # Thay đổi tiêu đề
    header_title: str = "🎙️ Vibe Podcasting"  # ← Sửa đây
    header_subtitle: str = "Generate Long-form Multi-speaker AI Podcasts with VibeVoice"  # ← Sửa đây
    
    # Thay đổi labels
    podcast_settings_label: str = "### 🎛️ Podcast Settings"  # ← Sửa đây
    speaker_selection_label: str = "### 🎭 Speaker Selection"  # ← Sửa đây
    
    # Thay đổi button text
    generate_btn: str = "🚀 Generate Podcast"  # ← Sửa đây
    random_example_btn: str = "🎲 Random Example"  # ← Sửa đây
```

#### Chỉnh sửa Slider và Input Settings:
```python
@dataclass
class UIConfig:
    # Số lượng speakers
    num_speakers_min: int = 1  # ← Sửa đây
    num_speakers_max: int = 4  # ← Sửa đây
    num_speakers_default: int = 2  # ← Sửa đây
    
    # Textbox size
    script_input_lines: int = 10  # ← Sửa đây
    prompt_output_lines: int = 25  # ← Sửa đây
```

#### Chỉnh sửa CSS và Theme:
```python
@dataclass
class UIConfig:
    # CSS tùy chỉnh
    custom_css: str = """.gradio-container { 
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; 
    }"""  # ← Thêm CSS của bạn ở đây
    
    # Theme
    theme: str = "Soft"  # ← Thay đổi theme: "Soft", "Default", "Monochrome", etc.
```

#### Chỉnh sửa Default Speakers:
```python
@dataclass
class UIConfig:
    default_speakers: List[str] = None
    
    def __post_init__(self):
        if self.default_speakers is None:
            # ← Thay đổi danh sách speakers mặc định ở đây
            self.default_speakers = ['en-Alice_woman', 'en-Carter_man', 'en-Frank_man', 'en-Maya_woman']
```

---

### 2. **File UI Components - `demo/colab_ui.py`** ⭐⭐

Đây là nơi chỉnh sửa **layout và cấu trúc** giao diện:

#### Chỉnh sửa Header:
```python
def create_header_html() -> str:
    """
    Tạo HTML header cho giao diện
    CHỈNH SỬA ĐÂY để thay đổi header
    """
    return f"""
    <div style="text-align: center; margin: 20px auto; max-width: 800px;">
        <h1 style="font-size: 2.5em; margin-bottom: 10px;">{config.ui.header_title}</h1>
        <!-- ← Thêm HTML tùy chỉnh của bạn ở đây -->
    </div>
    """
```

#### Chỉnh sửa Layout Settings Column:
```python
def create_settings_column(voice_manager: VoiceManager) -> tuple:
    """
    Tạo cột settings bên trái
    CHỈNH SỬA ĐÂY để thay đổi layout settings
    """
    with gr.Group():
        # ← Thêm/bớt/xóa các components ở đây
        gr.Markdown(config.ui.podcast_settings_label)
        # ...
```

#### Chỉnh sửa Layout Generation Column:
```python
def create_generation_column() -> tuple:
    """
    Tạo cột generation bên phải
    CHỈNH SỬA ĐÂY để thay đổi layout generation
    """
    with gr.Group():
        # ← Thêm/bớt/xóa các components ở đây
        script_input = gr.Textbox(...)
        # ...
```

#### Chỉnh sửa Usage Tips:
```python
def create_usage_tips_section(generator: PodcastGenerator) -> gr.Examples:
    """
    Tạo section usage tips và examples
    CHỈNH SỬA ĐÂY để thay đổi tips và examples
    """
    with gr.Accordion(config.ui.usage_tips_label, open=config.ui.usage_tips_accordion_open):
        gr.Markdown("""- **Upload Your Own Voices:** ...  
        - **Timestamps:** ...""")  # ← Sửa text tips ở đây
```

---

### 3. **File Prompt Builder UI - `demo/colab_prompt_builder.py`** ⭐⭐

Chỉnh sửa giao diện Prompt Builder:

#### Chỉnh sửa Prompt Format:
```python
def build_conversation_prompt(topic, *speaker_names):
    """
    Generate prompt for LLM to create podcast script
    CHỈNH SỬA ĐÂY để thay đổi format prompt
    """
    # ← Sửa format của prompt ở đây
    prompt = f"""
    You are a professional podcast scriptwriter. 
    ...
    """
    return prompt
```

#### Chỉnh sửa Prompt Builder Layout:
```python
def create_prompt_builder_ui():
    """
    Tạo giao diện Prompt Builder
    ĐÂY LÀ HÀM CHÍNH ĐỂ TẠO PROMPT BUILDER UI - CHỈNH SỬA ĐÂY ĐỂ THAY ĐỔI UI
    """
    with gr.Blocks(title="Prompt Builder") as demo:
        # ← Thêm/bớt/xóa components ở đây
        ...
```

---

## 🎨 Các Thay Đổi Phổ Biến

### Thay đổi màu sắc và styling:

1. **Trong `colab_config.py`**:
```python
custom_css: str = """
.gradio-container { 
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);  /* ← Thêm background */
}

/* Thêm CSS tùy chỉnh của bạn */
.generate-btn {
    background: linear-gradient(135deg, #059669 0%, #0d9488 100%);
    border-radius: 12px;
}
"""
```

### Thay đổi layout (2 cột → 3 cột, etc.):

1. **Trong `colab_ui.py`**, hàm `create_demo_interface()`:
```python
with gr.Row():
    with gr.Column(scale=1):  # ← Thay đổi scale hoặc thêm cột mới
        ...
    with gr.Column(scale=2):
        ...
    # Thêm cột thứ 3:
    with gr.Column(scale=1):
        ...
```

### Thêm components mới:

1. **Trong `colab_ui.py`**, thêm vào các hàm tạo UI:
```python
def create_settings_column(voice_manager: VoiceManager) -> tuple:
    with gr.Group():
        # Thêm component mới:
        new_slider = gr.Slider(
            minimum=0,
            maximum=100,
            value=50,
            label="New Setting"
        )
        # ...
```

---

## 📝 Checklist Chỉnh Sửa

- [ ] **Text và Labels**: Sửa trong `colab_config.py` → `UIConfig`
- [ ] **Layout và Components**: Sửa trong `colab_ui.py`
- [ ] **CSS và Styling**: Sửa trong `colab_config.py` → `UIConfig.custom_css`
- [ ] **Prompt Builder**: Sửa trong `colab_prompt_builder.py`
- [ ] **Default Values**: Sửa trong `colab_config.py`
- [ ] **Theme**: Sửa trong `colab_config.py` → `UIConfig.theme`

---

## 🔍 Tìm Kiếm Nhanh

| Muốn chỉnh sửa | File | Hàm/Class |
|----------------|------|-----------|
| Text, Labels, Buttons | `colab_config.py` | `UIConfig` |
| Header HTML | `colab_ui.py` | `create_header_html()` |
| Settings Layout | `colab_ui.py` | `create_settings_column()` |
| Generation Layout | `colab_ui.py` | `create_generation_column()` |
| CSS Styling | `colab_config.py` | `UIConfig.custom_css` |
| Prompt Builder UI | `colab_prompt_builder.py` | `create_prompt_builder_ui()` |
| Prompt Format | `colab_prompt_builder.py` | `build_conversation_prompt()` |

---

## 💡 Tips

1. **Bắt đầu từ Config**: Hầu hết các thay đổi đơn giản có thể làm trong `colab_config.py`
2. **Test từng thay đổi**: Thay đổi nhỏ, test, rồi tiếp tục
3. **Backup**: Lưu backup trước khi chỉnh sửa lớn
4. **Gradio Docs**: Tham khảo [Gradio Documentation](https://gradio.app/docs/) để biết thêm components

---

*Happy Customizing! 🎨*

