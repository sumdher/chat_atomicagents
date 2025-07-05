#!/bin/bash

MAGENTA='\033[1;35m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' 

cleanup() {
    echo
    echo "(bash)${MAGENTA}${BOLD}Stopping processes..."
    kill -TERM $client_pid 2>/dev/null
    kill -TERM $server_pid 2>/dev/null
    wait $client_pid 2>/dev/null
    wait $server_pid 2>/dev/null
    echo
    echo -e "(bash) ${MAGENTA}${BOLD}All processes stopped${NC}"
    echo
}

trap cleanup EXIT

LOCAL_MODE=0
if [[ "$1" == "local" ]]; then
    LOCAL_MODE=1
fi

# backend server
echo -e "(bash) ${MAGENTA}${BOLD}Starting backend server...${NC}"
cd server

if [[ $LOCAL_MODE -eq 1 ]]; then
    echo "(bash) ${MAGENTA}(with Docker DB)${NC}"
    LOCAL=1 python server.py &
else
    unset LOCAL
    python server.py &
fi
server_pid=$!

sleep 11
echo -e "(bash) ${MAGENTA}${BOLD}Backend server started${NC} ${YELLOW}(PID: $server_pid)${NC}"
echo

# frontend client
echo -e "(bash) ${MAGENTA}${BOLD}Starting Frontend client...${NC}"
cd ../client
npm run dev &
client_pid=$!
sleep 2
echo
echo -e "(bash) ${MAGENTA}${BOLD}Frontend client started at${NC} ${YELLOW}PID: $client_pid${MAGENTA}; Backend server at${NC} ${YELLOW}PID: $server_pid${NC}"
echo -e "(bash) ${MAGENTA}${BOLD}Press${NC} ${WHITE}${BOLD} CTRL + C  ${MAGENTA}${BOLD}to exit${NC}"
echo
echo
echo

wait $server_pid $client_pid
