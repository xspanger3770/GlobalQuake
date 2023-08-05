package globalquake.database_old;

public class SeedlinkNetwork {
	
	public static final int DISCONNECTED = 0;
	public static final int CONNECTING = 1;
	public static final int CONNECTED = 2;

	public int status;

	private final String host;
	private final String name;
	private final byte id;

	public int availableStations;
	public int selectedStations;
	public int connectedStations;

	public Thread seedlinkThread;
	
	public SeedlinkNetwork(byte id, String name, String host) {
		this.host = host;
		this.name = name;
		this.id = id;
	}

	public String getName() {
		return name;
	}

	public String getHost() {
		return host;
	}

	public byte getId() {
		return id;
	}

}
