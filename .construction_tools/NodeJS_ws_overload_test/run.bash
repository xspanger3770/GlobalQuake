#check if Node is installed
if ! [ -x "$(command -v node)" ]; then
echo -e '\033[0;31mError: Node.js is not installed.\033[0m' >&2
  exit 1
fi

#check if ./node_modules/ws exists
if [ ! -d "./node_modules/ws" ]; then
    npm install
fi


#amount of processes
#connections per process
#address

node ws_overload_test.js 20 1000 "ws://localhost:8081/realtime_events/v1"