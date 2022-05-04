package com.morce.globalquake.core;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;

import com.morce.globalquake.settings.Settings;

public class ZejfNetClient {

	private boolean connected;
	private boolean connecting;
	private boolean intendedClose;

	private ZejfNetStation station;
	private Thread thread;
	protected Socket socket;
	protected ObjectOutputStream outputStream;
	protected ObjectInputStream inputStream;

	public ZejfNetClient(ZejfNetStation station) {
		this.station = station;
	}

	public void connect() {
		thread = new Thread("ZejfNet Socket") {
			public void run() {
				while (true) {
					try {
						intendedClose = false;
						connected = false;
						connecting = true;
						System.out.println("Connecting to ZejfNet...");
						socket = new Socket();
						socket.setSoTimeout(10000);
						socket.connect(new InetSocketAddress(Settings.zejfSeisIP, Settings.zejfSeisPort));
						outputStream = new ObjectOutputStream(socket.getOutputStream());
						inputStream = new ObjectInputStream(socket.getInputStream());
						outputStream.writeUTF("realtime");
						outputStream.flush();
						connecting = false;
						connected = true;
						System.out.println("Listening for ZejfNet packets...");
						while (true) {
							String command = "";
							command = inputStream.readUTF();
							if (!command.isEmpty()) {
								try {
									if (command.startsWith("log")) {
										long time = inputStream.readLong();
										int value = inputStream.readInt();
										getStation().addRecord(new SimpleLog(time, value));
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
						System.out.println("ZejfNet Disconnected, waiting 10s...");
						connected = false;
						connecting = false;
						try {
							sleep(10000);
						} catch (InterruptedException e) {
							e.printStackTrace();
						}
					}
				}
			};
		};
		thread.start();
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
	}

	public boolean isConnected() {
		return connected;
	}

	public boolean isConnecting() {
		return connecting;
	}

	public ZejfNetStation getStation() {
		return station;
	}

}
