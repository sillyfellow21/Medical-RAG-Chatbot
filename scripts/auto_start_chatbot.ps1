$repoRoot = "E:\Bdtorrent\Medical-RAG-Chatbot\my-custom-chatbot"
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    exit 1
}

$streamlitPort = 8501
$active = Get-NetTCPConnection -LocalPort $streamlitPort -ErrorAction SilentlyContinue |
    Where-Object { $_.State -eq "Listen" }

if (-not $active) {
    Start-Process -FilePath $pythonExe `
        -ArgumentList "-m", "streamlit", "run", "streamlit_app.py", "--server.headless", "true", "--server.port", "$streamlitPort" `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden
}
