const cluster = require('cluster');
const WebSocket = require('ws')

const numProcesses = process.argv[2]; // Get number of processes from argument
const numConnectionsPerProcess = process.argv[3]; // Get number of connections per process from argument
const serverAddress = process.argv[4]; // Get server address from argument
const futureTime = Date.now() + 5000; // Set time 5 seconds in the future


if (cluster.isMaster) {
    console.log(`Master process started (PID: ${process.pid})`);
    console.log(`Starting ${numProcesses} processes, each creating ${numConnectionsPerProcess} connections to ${serverAddress}`);

  // Spawn worker processes
  for (let i = 0; i < numProcesses; i++) {
    cluster.fork();
  }

  cluster.on('exit', (worker, code, signal) => {
    console.log(`Worker process (PID: ${worker.process.pid}) exited with code: ${code}, signal: ${signal}`);
  });
} 

else {
  console.log(`Worker process started (PID: ${process.pid})`);

  // Wait until future time before creating connections
  while (Date.now() < futureTime) {}

  // Create WebSocket connections
  for (let i = 0; i < numConnectionsPerProcess; i++) {
    const ws = new WebSocket(serverAddress);
    ws.on('open', () => console.log(`Worker ${process.pid} - Connection ${i+1} opened`));
    ws.on('error', (error) => console.error(`Worker ${process.pid} - Connection ${i+1} error:`, error));
  }
}
