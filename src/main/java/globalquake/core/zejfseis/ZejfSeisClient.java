package globalquake.core.zejfseis;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;

import globalquake.core.SimpleLog;
import globalquake.main.Settings;

public class ZejfSeisClient {

	private static final int COMPATIBILITY_VERSION = 4;
	private boolean connected;
	private boolean connecting;
	private boolean intendedClose;
	private boolean reconnect;

	private ZejfSeisStation station;
	private Thread thread;
	protected Socket socket;
	protected OutputStream outputStream;
	protected InputStream inputStream;
	private int errVal;
	private int sampleRate;

	public ZejfSeisClient(ZejfSeisStation station) {
		this.station = station;
		init();
	}

	private void init() {
		thread = new Thread("ZejfNet Socket") {
			public void run() {
				do {
					reconnect = false;
					runZejfClient();
					if (!Settings.zejfSeisAutoReconnect && !reconnect) {
						break;
					}
					try {
						sleep(reconnect ? 100 : 10000);
					} catch (InterruptedException e) {
						break;
					}
				} while (Settings.zejfSeisAutoReconnect || reconnect);
			};
		};
	}

	public void connect() {
		if (!thread.isAlive()) {
			thread.start();
		}
	}

	public void reconnect() {
		if (!thread.isAlive()) {
			init();
			thread.start();
		} else {
			reconnect = true;
			disconnect();
		}
	}

	public void disconnect() {
		if (socket != null) {
			try {
				intendedClose = true;
				socket.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		thread.interrupt();
	}

	private void runZejfClient() {
		Thread heartbeatThread = null;
		try {
			intendedClose = false;
			connected = false;
			connecting = true;
			System.out.println("Connecting to ZejfNet " + Settings.zejfSeisIP + ":" + Settings.zejfSeisPort);
			socket = new Socket();
			socket.setSoTimeout(6000);
			socket.connect(new InetSocketAddress(Settings.zejfSeisIP, Settings.zejfSeisPort), 6000);
			outputStream = socket.getOutputStream();
			inputStream = socket.getInputStream();
			receiveInitialInfo();
			outputStream.flush();
			connecting = false;
			connected = true;

			heartbeatThread = new Thread("Heartbeat") {
				public void run() {
					while (true) {
						try {
							sleep(5000);
						} catch (InterruptedException e) {
							break;
						}
						try {
							outputStream.write("heartbeat\n".getBytes());
							outputStream.flush();
						} catch (Exception e) {
							e.printStackTrace();
						}
					}
				};
			};

			heartbeatThread.start();

			System.out.println("Listening for ZejfNet packets...");
			while (true) {
				String command = "";
				command = readString();
				if (!command.isEmpty()) {
					try {
						if (command.startsWith("realtime")) {
							int value;
							while((value = Integer.valueOf(readString())) != errVal) {
								long time = Long.valueOf(readString()) * (1000 / sampleRate);
								getStation().addRecord(new SimpleLog(time, value));
							}
						}
					} catch (Exception e) {
						System.err.println("Unable to parse command '" + command + "': " + e.getMessage());
						break;
					}
				}
			}
			socket.close();
		} catch (IOException e) {
			System.err.println("ZejfNet Disconnected: " + e.getMessage());
			if (!intendedClose) {
				e.printStackTrace();
			}
		} finally {
			System.out.println("ZejfNet Disconnected");
			connected = false;
			connecting = false;
			if (heartbeatThread != null)
				heartbeatThread.interrupt();
		}
	}
	
	private void receiveInitialInfo() throws IOException, NumberFormatException {
		String compat_version = readString();
		System.out.println(compat_version);
		int comp = Integer.valueOf(compat_version.split(":")[1]);
		if (comp != COMPATIBILITY_VERSION) {
			System.err.println("ZEJFNET INCOMPATIBLE");
			return;
		} else {
			System.out.println("Compatibility numbers matched");
		}
		String sample_rate = readString();
		String err_value = readString();
		String last_log_id = readString();

		System.out.println(sample_rate);
		System.out.println(err_value);
		System.out.println(last_log_id);

		int sr = Integer.valueOf(sample_rate.split(":")[1]);
		int err = Integer.valueOf(err_value.split(":")[1]);
		long lli = Long.valueOf(last_log_id.split(":")[1]);
		
		errVal = err;
		sampleRate = sr;
		station.getAnalysis().setSampleRate(sampleRate);

		System.out.println("Received info: ");
		System.out.println("Sample rate: " + sr + "sps");
		System.out.println("Error value: " + err);
		System.out.println("Last log id: " + lli);
		
		outputStream.write(("realtime\n"+lli+"\n").getBytes());
	}

	private String readString() throws IOException {
		StringBuilder result = new StringBuilder();
		while (true) {
			char ch = (char) socket.getInputStream().read();
			if (ch == '\n') {
				break;
			} else {
				result.append(ch);
			}
		}
		return result.toString();
	}

	public boolean isConnected() {
		return connected;
	}

	public boolean isConnecting() {
		return connecting;
	}

	public ZejfSeisStation getStation() {
		return station;
	}

}
