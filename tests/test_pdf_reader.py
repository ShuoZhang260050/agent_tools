import pymupdf

from agent.tools.pdf_reader import read_pdf


def _make_pdf(path, pages_text):
    doc = pymupdf.open()
    for t in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), t, fontsize=12)
    doc.save(str(path))
    doc.close()


def test_read_pdf_basic(tmp_path):
    p = tmp_path / "t.pdf"
    _make_pdf(p, ["Hello World", "Second Page"])
    out = read_pdf.invoke({"file_path": str(p)})
    assert "第 1 页" in out
    assert "Hello World" in out
    assert "Second Page" in out
    assert "共 2 页" in out


def test_read_pdf_not_found():
    out = read_pdf.invoke({"file_path": "nonexistent.pdf"})
    assert "文件不存在" in out


def test_read_pdf_not_pdf(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("hi", encoding="utf-8")
    out = read_pdf.invoke({"file_path": str(p)})
    assert "不是 PDF" in out


def test_read_pdf_page_range(tmp_path):
    p = tmp_path / "t.pdf"
    _make_pdf(p, ["AAA", "BBB", "CCC"])
    out = read_pdf.invoke({"file_path": str(p), "page_start": 2, "page_end": 3})
    assert "第 2 页" in out
    assert "第 3 页" in out
    assert "第 1 页" not in out


def test_read_pdf_max_chars_truncation(tmp_path):
    p = tmp_path / "t.pdf"
    _make_pdf(p, ["Hello", "World", "Again"])
    out = read_pdf.invoke({"file_path": str(p), "max_chars": 8})
    assert "Hello" in out
    assert "继续读取请用 page_start=2" in out
    assert "World" not in out


def test_read_pdf_finishes_all_pages_no_continue_hint(tmp_path):
    p = tmp_path / "t.pdf"
    _make_pdf(p, ["Only Page"])
    out = read_pdf.invoke({"file_path": str(p)})
    assert "第 1 页" in out
    assert "Only Page" in out
    assert "继续读取请用 page_start" not in out
