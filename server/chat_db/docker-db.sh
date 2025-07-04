# docker-db.sh
#!/bin/bash

CONTAINER_NAME="chat-app-db"
DB_USER="chat_user"
DB_PASSWORD="chat_password"
DB_NAME="chat_app"
VOLUME_NAME="chat_app_data"

# Get the absolute path of the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if container exists
if [ ! "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
    echo "Creating new PostgreSQL container..."
    docker run -d --name $CONTAINER_NAME \
      -e POSTGRES_USER=$DB_USER \
      -e POSTGRES_PASSWORD=$DB_PASSWORD \
      -e POSTGRES_DB=$DB_NAME \
      -p 9247:5432 \
      -v $VOLUME_NAME:/var/lib/postgresql/data \
      postgres:15

else
    # Start existing container if not running
    if [ ! "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        echo "Starting existing PostgreSQL container..."
        docker start $CONTAINER_NAME
    else
        echo "PostgreSQL container is already running"
    fi
fi

# Wait for PostgreSQL to start...
echo "Waiting for PostgreSQL to start..."
while ! docker exec $CONTAINER_NAME pg_isready -U $DB_USER -d $DB_NAME -h localhost > /dev/null 2>&1; do
  sleep 1
done

echo "PostgreSQL is ready!"