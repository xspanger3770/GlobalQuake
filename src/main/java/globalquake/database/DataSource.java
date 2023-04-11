package globalquake.database;

public class DataSource {

	private String url;
	private String name;
	private byte id;

	public DataSource(String name, String fdsnwsURL, byte id) {
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
