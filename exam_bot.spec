# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置

    pyinstaller --noconfirm --clean exam_bot.spec

产物：dist/自动答题/自动答题.exe（文件夹版，双击运行）
想改成单文件：把 COLLECT(...) 换成 EXE(..., a.binaries, a.datas, ...) 即可 ——
但单文件每次启动都要把几百 MB 解压到临时目录，启动会慢很多，不建议。
"""
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = [], [], []

# onnxruntime / rapidocr 这类包带 DLL 和模型文件，必须整包收进来，少一个文件就崩
for pkg in ("rapidocr_onnxruntime", "onnxruntime", "pyclipper", "shapely",
            "pyautogui", "PIL"):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    except Exception:
        continue
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# 本机要是装了 paddleocr，也一起打进去当兜底（没装就是空的）
for pkg in ("paddle", "paddleocr"):
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass

hiddenimports += ["cv2", "requests"]

a = Analysis(
    ["exam_bot_gui.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "scipy", "pandas", "IPython", "pytest", "notebook",
              "torch", "tensorflow"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="自动答题",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,       # 保留控制台：GUI 启动后自己把它藏起来，
                        # 但 --selftest 和崩溃信息还能看得到
    icon="app.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="自动答题",
)