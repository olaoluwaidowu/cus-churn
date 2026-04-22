# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the Churn Predictor desktop app.
Run from the cus-churn/desktop/ directory:
    pyinstaller build.spec
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Pull in all customtkinter assets (themes, images, fonts)
ctk_datas = collect_data_files("customtkinter")

# Model file — one level up from desktop/
model_src = os.path.join("..", "model", "rf_pipeline.pkl")

a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=[],
    datas=[
        (model_src, "model"),          # bundle the trained model
        *ctk_datas,                    # customtkinter themes & assets
    ],
    hiddenimports=[
        *collect_submodules("sklearn"),
        *collect_submodules("customtkinter"),
        "joblib",
        "pandas",
        "numpy",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "IPython", "jupyter", "notebook"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ChurnPredictor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # no console window — GUI only
    icon=None,              # add an .ico file path here if desired
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ChurnPredictor",
)
