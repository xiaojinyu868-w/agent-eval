"""官方任务指令导入器：支持 xlsx/txt/json，零第三方依赖。"""

import json
import os
import re
import zipfile
import xml.etree.ElementTree as ET

NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def _col_idx(cell_ref: str) -> int:
    letters = re.sub(r"\d", "", cell_ref or "")
    idx = 0
    for ch in letters:
        idx = idx * 26 + ord(ch.upper()) - ord("A") + 1
    return max(idx - 1, 0)


def _read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values = []
    for si in root.findall("m:si", NS):
        values.append("".join(t.text or "" for t in si.findall(".//m:t", NS)))
    return values


def _sheet_paths(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    sheets = []
    for sh in wb.findall("m:sheets/m:sheet", NS):
        name = sh.attrib.get("name", "Sheet")
        rid = sh.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        target = relmap.get(rid, "")
        if target:
            sheets.append((name, "xl/" + target.lstrip("/")))
    return sheets


def _rows_from_sheet(zf: zipfile.ZipFile, path: str, shared_strings: list[str]) -> list[list[str]]:
    root = ET.fromstring(zf.read(path))
    rows = []
    for row in root.findall(".//m:sheetData/m:row", NS):
        values = []
        for c in row.findall("m:c", NS):
            idx = _col_idx(c.attrib.get("r", ""))
            while len(values) < idx:
                values.append("")
            v = c.find("m:v", NS)
            value = "" if v is None or v.text is None else v.text
            if c.attrib.get("t") == "s" and value:
                value = shared_strings[int(value)]
            elif c.attrib.get("t") == "inlineStr":
                value = "".join(t.text or "" for t in c.findall(".//m:t", NS))
            values.append(value)
        rows.append(values)
    return rows


def load_xlsx_instructions(path: str) -> list[dict]:
    records = []
    with zipfile.ZipFile(path) as zf:
        shared_strings = _read_shared_strings(zf)
        for sheet_name, sheet_path in _sheet_paths(zf):
            rows = _rows_from_sheet(zf, sheet_path, shared_strings)
            if not rows:
                continue
            header = [str(x).strip() for x in rows[0]]
            id_col = next((i for i, h in enumerate(header) if h.lower() in ("id", "编号", "任务id")), 0)
            text_col = next((i for i, h in enumerate(header) if "指令" in h or "instruction" in h.lower()), 1 if len(header) > 1 else 0)
            for n, row in enumerate(rows[1:], start=1):
                if text_col >= len(row):
                    continue
                text = str(row[text_col]).strip()
                if not text:
                    continue
                raw_id = str(row[id_col]).strip() if id_col < len(row) and str(row[id_col]).strip() else str(n)
                records.append({
                    "id": raw_id,
                    "source": os.path.basename(path),
                    "sheet": sheet_name,
                    "instruction": text,
                })
    return records


def load_instruction_records(path: str | None, default_instruction: str | None = None) -> list[dict]:
    if not path:
        return [{"id": "default", "source": "builtin", "instruction": default_instruction or ""}]
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xlsx":
        return load_xlsx_instructions(path)
    if ext == ".txt" or ext == ".md":
        with open(path, "r", encoding="utf-8") as f:
            return [{"id": os.path.splitext(os.path.basename(path))[0], "source": os.path.basename(path), "instruction": f.read()}]
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [dict(x) for x in data if isinstance(x, dict) and x.get("instruction")]
        if isinstance(data, dict) and data.get("instructions"):
            return [dict(x) for x in data["instructions"] if isinstance(x, dict) and x.get("instruction")]
        if isinstance(data, dict) and data.get("instruction"):
            return [{"id": data.get("id", "json"), "source": os.path.basename(path), "instruction": data["instruction"]}]
    raise ValueError(f"不支持的指令文件: {path}")


def filter_instruction_records(records: list[dict], instruction_id: str | None) -> list[dict]:
    if not instruction_id:
        return records
    wanted = {x.strip() for x in instruction_id.split(",") if x.strip()}
    return [r for r in records if str(r.get("id")) in wanted]
