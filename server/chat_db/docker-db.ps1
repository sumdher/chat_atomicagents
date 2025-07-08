# $Container = "chat-app-db"
# $DbUser    = "chat_user"
# $DbPass    = "chat_password"
# $DbName    = "chat_app"
# $Volume    = "chat_app_data"

# $cid = docker ps -a -q -f "name=$Container"
# if (-not $cid) {
#     Write-Host "Creating new PostgreSQL container..."
#     docker run -d --name $Container `
#         -e "POSTGRES_USER=$DbUser" `
#         -e "POSTGRES_PASSWORD=$DbPass" `
#         -e "POSTGRES_DB=$DbName" `
#         -p 9247:5432 `
#         -v "$Volume:/var/lib/postgresql/data" `
#         postgres:15
# } else {
#     $running = docker ps -q -f "name=$Container"
#     if (-not $running) {
#         Write-Host "Starting existing PostgreSQL container..."
#         docker start $Container
#     } else {
#         Write-Host "PostgreSQL container is already running."
#     }
# }

# Write-Host "Waiting for PostgreSQL to start..."
# while (-not (docker exec $Container pg_isready -U $DbUser -d $DbName -h localhost)) {
#     Start-Sleep -Seconds 1
# }
# Write-Host "PostgreSQL is ready!"

# docker-db.ps1 - Windows-Specific Version
$ErrorActionPreference = "Stop"

$Container = "chat-app-db"
$DbUser = "chat_user"
$DbPass = "chat_password"
$DbName = "chat_app"
$Volume = "chat_app_data"

# Check if Docker is running
try {
    docker ps | Out-Null
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
}

# Container management
try {
    $cid = docker ps -a -q -f "name=$Container"
    
    if (-not $cid) {
        Write-Host "Creating new PostgreSQL container..."
        docker run -d --name $Container `
            -e "POSTGRES_USER=$DbUser" `
            -e "POSTGRES_PASSWORD=$DbPass" `
            -e "POSTGRES_DB=$DbName" `
            -p 9247:5432 `
            -v "${Volume}:/var/lib/postgresql/data" `
            --restart unless-stopped `
            postgres:15
    }
    else {
        $running = docker ps -q -f "name=$Container"
        if (-not $running) {
            Write-Host "Starting existing PostgreSQL container..."
            docker start $Container | Out-Null
        }
        else {
            Write-Host "PostgreSQL container is already running."
        }
    }

    # Wait for PostgreSQL with timeout
    Write-Host "Waiting for PostgreSQL to start (max 60 seconds)..."
    $timeout = 60
    $startTime = Get-Date
    
    while (-not (docker exec $Container pg_isready -U $DbUser -d $DbName -h localhost)) {
        if ((Get-Date) - $startTime -gt [TimeSpan]::FromSeconds($timeout)) {
            throw "Timed out waiting for PostgreSQL to start"
        }
        Start-Sleep -Seconds 1
    }
    
    Write-Host "✅ PostgreSQL is ready!"
    exit 0
}
catch {
    Write-Host "❌ Error: $_"
    exit 1
}