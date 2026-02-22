#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清空以下目录内容：
  /home/kzlab/muse/Savvy/Data_collection/output/2_paper_processed
  /home/kzlab/muse/Savvy/Data_collection/output/3_pdf
  /home/kzlab/muse/Savvy/Data_collection/output/4_latex
  /home/kzlab/muse/Savvy/Data_collection/output/final_dataset
并将：
  /home/kzlab/muse/Savvy/Data_collection/state/arxiv_state.json
写为一个空对象 {}。
"""

from pathlib import Path
import shutil
import sys

# 需清空的目录
DIRS = [
    Path("/home/kzlab/muse/Savvy/Data_collection/output/2_paper_processed"),
    Path("/home/kzlab/muse/Savvy/Data_collection/output/3_pdf"),
    Path("/home/kzlab/muse/Savvy/Data_collection/output/4_latex"),
    Path("/home/kzlab/muse/Savvy/Data_collection/output/final_dataset"),
]

# JSON 文件路径
JSON = Path("/home/kzlab/muse/Savvy/Data_collection/state/arxiv_state.json")

# 目录安全保护根（只有在该根路径下的目录才会被清空）
SAFE_ROOT = Path("/home/kzlab/muse/Savvy/Data_collection/output").resolve()

def under_safe_root(p: Path) -> bool:
    try:
        p.resolve().relative_to(SAFE_ROOT)
        return True
    except Exception:
        return False

def clear_dir(d: Path):
    if not d.exists():
        print(f"[skip] 目录不存在：{d}")
        return
    if not d.is_dir():
        print(f"[skip] 非目录路径：{d}")
        return
    # 安全保护：必须在 SAFE_ROOT 之下
    if not under_safe_root(d):
        print(f"[ABORT] 安全保护生效：{d} 不在 {SAFE_ROOT} 下，已跳过。")
        return

    errors = 0
    for child in d.iterdir():
        try:
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
            else:
                # 其他非常见文件类型（如设备文件），尝试直接删除
                child.unlink(missing_ok=True)
        except Exception as e:
            errors += 1
            print(f"[error] 删除失败：{child} -> {e}", file=sys.stderr)
    if errors == 0:
        print(f"[ok] 已清空目录：{d}")
    else:
        print(f"[warn] 目录清空完成（存在 {errors} 个删除错误）：{d}")

def clear_json(p: Path, write_empty_object: bool = True):
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with p.open("w", encoding="utf-8") as f:
            if write_empty_object:
                f.write("{}\n")  # 保持为合法 JSON
            else:
                f.truncate(0)    # 置空文件（可能不是合法 JSON）
        print(f"[ok] 已清空文件：{p}（写入 {'{}' if write_empty_object else '空文件'}）")
    except Exception as e:
        print(f"[error] 清空 JSON 失败：{p} -> {e}", file=sys.stderr)

def main():
    for d in DIRS:
        clear_dir(d)
    clear_json(JSON, write_empty_object=True)

if __name__ == "__main__":
    main()
