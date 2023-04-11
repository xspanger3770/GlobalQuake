package globalquake.core.report;

import java.io.Serializable;

public class StationReport implements Serializable{

	private static final long serialVersionUID = -8686117122281460600L;
	public String networkCode;
	public String stationCode;
	public String channelName;
	public String locationCode;
	public double lat;
	public double lon;
	public double alt;

	public StationReport(String networkCode, String stationCode, String channelName, String locationCode, double lat,
			double lon, double alt) {
		super();
		this.networkCode = networkCode;
		this.stationCode = stationCode;
		this.channelName = channelName;
		this.locationCode = locationCode;
		this.lat = lat;
		this.lon = lon;
		this.alt = alt;
	}

}
