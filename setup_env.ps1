 # HyperFrames 环境配置脚本
 # 每次打开新终端运行此脚本以加载环境变量
 # 使用方法: . .\setup_env.ps1

 Write-Host "正在配置 HyperFrames 环境..." -ForegroundColor Cyan

 # 1. 添加 FFmpeg 到 PATH
 $ffmpegDir = "D:\PythonProject\Research\.ffmpeg\bin"
 if (Test-Path $ffmpegDir) {
     $env:PATH = "$ffmpegDir;$env:PATH"
     $ver = & ffmpeg -version 2>&1 | Select-String "ffmpeg version"
     if ($ver) {
         Write-Host "  ✓ FFmpeg: $($ver.Matches.Value)" -ForegroundColor Green
     }
 } else {
     Write-Host "  ✗ FFmpeg not found at $ffmpegDir" -ForegroundColor Red
 }

 # 2. npm registry 配置
 cmd.exe /c "npm config set registry https://registry.npmmirror.com" 2>$null
 Write-Host "  ✓ npm registry: https://registry.npmmirror.com" -ForegroundColor Green

 # 3. 验证环境
 Write-Host ""
 Write-Host "环境检查:" -ForegroundColor Cyan
 $nodeVer = node --version
 Write-Host "  Node.js: $nodeVer" -ForegroundColor $(if ($nodeVer -ge "v22") { "Green" } else { "Yellow" })

 try {
     $hfVer = cmd.exe /c "npx --yes hyperframes --version" 2>&1 | Select-Object -First 1
     Write-Host "  HyperFrames CLI: v$hfVer" -ForegroundColor Green
 } catch {
     Write-Host "  HyperFrames CLI: 未安装" -ForegroundColor Yellow
 }

 Write-Host ""
 Write-Host "快速开始:" -ForegroundColor Cyan
 Write-Host "  cd my-video"
 Write-Host "  npm run dev     # 启动预览服务器"
 Write-Host "  npm run check   # 检查组合"
 Write-Host "  npm run render  # 渲染为 MP4"
 Write-Host ""
 Write-Host "官方文档: https://hyperframes.heygen.com" -ForegroundColor Gray
