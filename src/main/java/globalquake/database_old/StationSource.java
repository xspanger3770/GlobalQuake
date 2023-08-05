package globalquake.database_old;

public class StationSource {

	private final String url;
	private final String name;
	private final byte id;

	public StationSource(String name, String fdsnwsURL, byte id) {
		this.url = fdsnwsURL;
		this.name = name;
		this.id = id;
	}

	public String getUrl() {
		return url;
	}

	public String getName() {
		return name;
	}

	public byte getId() {
		return id;
	}

}
