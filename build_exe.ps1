# 打包成 exe。用法：在 PowerShell 里进入本目录，执行  .\build_exe.ps1
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

Write-Host "== 1/3 检查依赖 ==" -ForegroundColor Cyan
python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "没装 PyInstaller，正在安装..." -ForegroundColor Yellow
    python -m pip install pyinstaller
}
python -c "import rapidocr_onnxruntime, pyautogui, cv2, numpy, requests, PIL" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "缺少运行依赖，正在安装..." -ForegroundColor Yellow
    python -m pip install rapidocr-onnxruntime pyautogui opencv-python requests numpy pillow
}

Write-Host "== 2/3 开始打包（第一次要几分钟，请耐心等）==" -ForegroundColor Cyan
python -m PyInstaller --noconfirm --clean --distpath dist --workpath build exam_bot.spec

# --clean 会把 dist\自动答题 整个删掉重建，所以说明文件要在这里补回去
$outDir = Join-Path $PSScriptRoot "dist\自动答题"
if (Test-Path -LiteralPath (Join-Path $PSScriptRoot "使用说明.txt")) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "使用说明.txt") -Destination $outDir -Force
}

Write-Host ""
Write-Host "== 3/3 打包后自检 ==" -ForegroundColor Cyan
& (Join-Path $outDir "自动答题.exe") --selftest

Write-Host ""
Write-Host "打包完成，产物目录：$outDir" -ForegroundColor Green
Write-Host "自检命令：& '$PSScriptRoot\dist\自动答题\自动答题.exe' --selftest" -ForegroundColor Green
