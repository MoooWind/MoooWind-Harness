from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml.ns import qn

OUTPUT_DIR = Path(__file__).resolve().parent / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def create_word(content):
    try:
        doc = Document()
        doc.styles['Normal'].font.name = 'Microsoft YaHei'
        doc.styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
        doc.styles['Normal'].font.size = Pt(11)

        lines = content.splitlines()
        title_index = next(
            (index for index, line in enumerate(lines) if line.strip().startswith("# ")),
            None,
        )
        document_title = (
            lines[title_index].strip()[2:].strip()
            if title_index is not None
            else "MoooWind 智能报告"
        )
        title = doc.add_heading('', level=0)
        run = title.add_run(document_title)
        run.font.color.rgb = RGBColor(0, 51, 153)
        run.font.name = 'Microsoft YaHei'
        run.font.size = Pt(22)
        run.bold = True

        for index, line in enumerate(lines):
            line = line.strip()
            if not line or index == title_index:
                continue
            
            if line.startswith("# "):
                doc.add_heading(line[2:], level=1)
            elif line.startswith("### "):
                doc.add_heading(line[4:], level=3)
            elif line.startswith("## "):
                doc.add_heading(line[3:], level=2)
            elif line.startswith("- ") or line.startswith("* "):
                doc.add_paragraph(line[2:], style="List Bullet")
            elif line.startswith(tuple(f"{number}. " for number in range(1, 10))):
                doc.add_paragraph(line.split(". ", 1)[1], style="List Number")
            else:
                p = doc.add_paragraph(line)
                p.paragraph_format.first_line_indent = Pt(22) 
                p.paragraph_format.line_spacing = 1.5 

        path = OUTPUT_DIR / "MoooWind_Word.docx"
        doc.save(path)
        return str(path)
    except Exception as error:
        raise RuntimeError(f"生成 Word 失败：{error}") from error

def create_excel(csv_data):
    try:
        lines = [line for line in csv_data.strip().split('\n') if line.strip()]
        data = [line.split(',') for line in lines]
        df = pd.DataFrame(data[1:], columns=data[0])
        path = OUTPUT_DIR / "MoooWind_Excel.xlsx"
        df.to_excel(path, index=False)
        return str(path)
    except Exception as e: 
        return f"生成Excel失败: {e}"

# 🚨 新增：读取本地文件工具
def read_local_file(file_path):
    """读取本地文本文件内容"""
    try:
        # 清理掉用户可能带上的引号
        file_path = file_path.strip().strip('"').strip("'")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if len(content) > 3000:
            return content[:3000] + "\n...(内容过长已截断)"
        return content
    except Exception as error:
        raise RuntimeError(f"读取文件失败：{error}") from error