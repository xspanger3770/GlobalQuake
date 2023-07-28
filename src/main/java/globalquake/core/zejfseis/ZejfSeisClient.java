package globalquake.core.zejfseis;

import globalquake.core.SimpleLog;
import globalquake.main.Settings;
import org.tinylog.Logger;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.Timer;
import java.util.TimerTask;

public class ZejfSeisClient {

	private static final int COMPATIBILITY_VERSION = 4;
	private boolean connected;
	private boolean connecting;
	private boolean intendedClose;
	private boolean reconnect;

	private final ZejfSeisStation station;
	private Thread thread;
	protected Socket socket;
	protected OutputStream outputStream;
	private int errVal;
	private int sampleRate;

	public ZejfSeisClient(ZejfSeisStation station) {
		this.station = station;
		init();
	}

	private void init() {
		thread = new Thread("ZejfNet Socket") {
			@SuppressWarnings("BusyWait")
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
			}
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
				Logger.error(e);
			}
		}
		thread.interrupt();
	}

	private void runZejfClient() {
		try {
			intendedClose = false;
			connected = false;
			connecting = true;
			System.out.println("Connecting to ZejfNet " + Settings.zejfSeisIP + ":" + Settings.zejfSeisPort);
			socket = new Socket();
			socket.setSoTimeout(6000);
			socket.connect(new InetSocketAddress(Settings.zejfSeisIP, Settings.zejfSeisPort), 6000);
			outputStream = socket.getOutputStream();
			receiveInitialInfo();
			outputStream.flush();
			connecting = false;
			connected = true;

			Timer timer = new Timer();

			timer.scheduleAtFixedRate(new TimerTask() {
				public void run() {
					try {
						outputStream.write("heartbeat\n".getBytes());
						outputStream.flush();
					} catch (Exception e) {
						Logger.error(e);
					}
				}
			}, 0, 5000);

			System.out.println("Listening for ZejfNet packets...");
			while (true) {
				String command;
				command = readString();
				if (!command.isEmpty()) {
					try {
						if (command.startsWith("realtime")) {
							int value;
							while((value = Integer.parseInt(readString())) != errVal) {
								long time = Long.parseLong(readString()) * (1000 / sampleRate);
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
				Logger.error(e);
			}
		} finally {
			System.out.println("ZejfNet Disconnected");
			connected = false;
			connecting = false;
        }
	}
	
	private void receiveInitialInfo() throws IOException, NumberFormatException {
		String compat_version = readString();
		System.out.println(compat_version);
		int comp = Integer.parseInt(compat_version.split(":")[1]);
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

		int sr = Integer.parseInt(sample_rate.split(":")[1]);
		int err = Integer.parseInt(err_value.split(":")[1]);
		long lli = Long.parseLong(last_log_id.split(":")[1]);
		
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
