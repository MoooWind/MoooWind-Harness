import re
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

OUTPUT_DIR = Path(__file__).resolve().parent / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def create_apple_style_ppt(title, outlines):
    try:
        prs = Presentation()
        prs.slide_width = Inches(10)
        prs.slide_height = Inches(5.625)
        blank_slide_layout = prs.slide_layouts[6]

        def add_textbox(slide, text, top, size, color, bold=True, align=PP_ALIGN.CENTER, height=2):
            txBox = slide.shapes.add_textbox(Inches(1), Inches(top), Inches(8), Inches(height))
            tf = txBox.text_frame
            tf.word_wrap = True
            tf.margin_left = Inches(0.08)
            tf.margin_right = Inches(0.08)
            p = tf.paragraphs[0]
            p.text = text
            p.alignment = align
            font = p.font
            font.name = 'Microsoft YaHei'
            font.size = Pt(size)
            font.bold = bold
            font.color.rgb = color

        def add_bullet_list(slide, points):
            text_box = slide.shapes.add_textbox(Inches(1.25), Inches(2.55), Inches(7.5), Inches(2.2))
            text_frame = text_box.text_frame
            text_frame.clear()
            text_frame.word_wrap = True
            text_frame.margin_left = Inches(0.12)
            text_frame.margin_right = Inches(0.12)
            for index, point in enumerate(points[:5]):
                paragraph = text_frame.paragraphs[0] if index == 0 else text_frame.add_paragraph()
                paragraph.text = f"- {point}"
                paragraph.level = 0
                paragraph.font.name = "Microsoft YaHei"
                paragraph.font.size = Pt(19)
                paragraph.font.color.rgb = RGBColor(210, 210, 210)
                paragraph.space_after = Pt(12)

        def create_base_slide(is_cover=False, page_num=1, total_pages=1):
            slide = prs.slides.add_slide(blank_slide_layout)
            background = slide.background
            fill = background.fill
            fill.solid()
            fill.fore_color.rgb = RGBColor(0, 0, 0)
            
            if not is_cover:
                add_textbox(
                    slide,
                    "MoooWind AI PRESENTATION",
                    0.45,
                    10,
                    RGBColor(100, 100, 100),
                    bold=False,
                    align=PP_ALIGN.LEFT,
                    height=0.3,
                )
                add_textbox(
                    slide,
                    f"{page_num} / {total_pages}",
                    5.05,
                    10,
                    RGBColor(100, 100, 100),
                    bold=False,
                    align=PP_ALIGN.RIGHT,
                    height=0.3,
                )
            return slide

        final_outlines = [outline.strip() for outline in outlines if outline.strip()]
        if not final_outlines:
            raise ValueError("PPT 至少需要一条有效的大纲")

        total_pages = len(final_outlines) + 1 

        # 封面
        slide = create_base_slide(is_cover=True)
        add_textbox(slide, title if title else "MoooWind 报告", 2.0, 48, RGBColor(255, 255, 255))
        add_textbox(
            slide,
            "由本地大模型自动生成",
            3.5,
            16,
            RGBColor(150, 150, 150),
            bold=False,
            height=0.5,
        )

        # 内容页
        for idx, outline in enumerate(final_outlines):
            page_num = idx + 2
            slide = create_base_slide(is_cover=False, page_num=page_num, total_pages=total_pages)
            
            clean_outline = outline.replace("大纲：", "", 1).strip()
            if "|" in clean_outline:
                slide_title, body = clean_outline.split("|", 1)
            else:
                parts = re.split(r"[:：]\s*", clean_outline, maxsplit=1)
                slide_title = parts[0]
                body = parts[1] if len(parts) > 1 else ""

            add_textbox(
                slide,
                slide_title.strip(),
                1.65,
                32,
                RGBColor(255, 255, 255),
                height=0.8,
            )
            points = [
                point.strip(" •-")
                for point in re.split(r"[；;\n]+|(?<=[。！？])\s*", body)
                if point.strip(" •-")
            ]
            if points:
                add_bullet_list(slide, points)

        path = OUTPUT_DIR / "MoooWind_AppleStyle.pptx"
        prs.save(path)
        return str(path)
    except Exception as error:
        raise RuntimeError(f"生成 PPT 失败：{error}") from error