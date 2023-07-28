package globalquake.database;

public class StationSource {

	private String url;
	private String name;
	private byte id;

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
