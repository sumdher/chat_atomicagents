$Container = "chat-app-db"
$DbUser    = "chat_user"
$DbPass    = "chat_password"
$DbName    = "chat_app"
$Volume    = "chat_app_data"

$cid = docker ps -a -q -f "name=$Container"
if (-not $cid) {
    Write-Host "Creating new PostgreSQL container..."
    docker run -d --name $Container `
        -e "POSTGRES_USER=$DbUser" `
        -e "POSTGRES_PASSWORD=$DbPass" `
        -e "POSTGRES_DB=$DbName" `
        -p 9247:5432 `
        -v "$Volume:/var/lib/postgresql/data" `
        postgres:15
} else {
    $running = docker ps -q -f "name=$Container"
    if (-not $running) {
        Write-Host "Starting existing PostgreSQL container..."
        docker start $Container
    } else {
        Write-Host "PostgreSQL container is already running."
    }
}

Write-Host "Waiting for PostgreSQL to start..."
while (-not (docker exec $Container pg_isready -U $DbUser -d $DbName -h localhost)) {
    Start-Sleep -Seconds 1
}
Write-Host "PostgreSQL is ready!"