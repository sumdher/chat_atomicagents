# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

$MAGENTA = "`e[1;35m"
$YELLOW = "`e[1;33m"
$BOLD = "`e[1m"
$NC = "`e[0m"
$WHITE = "`e[1;37m"

function Cleanup {
    Write-Host ""
    Write-Host "(pwsh)$MAGENTA${BOLD}Stopping processes...$NC"
    
    if ($serverJob) { Stop-Job $serverJob }
    if ($clientJob) { Stop-Job $clientJob }
    
    Get-Job | Remove-Job -Force
    
    Write-Host ""
    Write-Host "(pwsh) $MAGENTA${BOLD}All processes stopped$NC"
}

[Console]::TreatControlCAsInput = $false
[Console]::CancelKeyPress.Add({
    Cleanup
    exit
})

Write-Host "(pwsh) $MAGENTA${BOLD}Starting backend server...$NC"
$serverJob = Start-Job -ScriptBlock {
    Set-Location server
    python server.py
}

Start-Sleep -Seconds 11
Write-Host "(pwsh) $MAGENTA${BOLD}Backend server started$NC $YELLOW(PID: $($serverJob.Id))$NC"
Write-Host ""

Write-Host "(pwsh) $MAGENTA${BOLD}Starting Frontend client...$NC"
$clientJob = Start-Job -ScriptBlock {
    Set-Location client
    npm run dev
}
Start-Sleep -Seconds 4
Write-Host ""
Write-Host "(pwsh) $MAGENTA${BOLD}Frontend client started at$NC $YELLOW(PID: $($clientJob.Id))$MAGENTA${BOLD}; Backend server at$NC $YELLOW(PID: $($serverJob.Id))$NC"
Write-Host "(pwsh) $MAGENTA${BOLD}Press$NC $WHITE${BOLD} CTRL + C  $MAGENTA${BOLD}to exit$NC"
Write-Host ""
Write-Host ""
Write-Host ""

try {
    Wait-Job -Job $serverJob, $clientJob -ErrorAction SilentlyContinue | Out-Null
}
finally {
    Cleanup
}