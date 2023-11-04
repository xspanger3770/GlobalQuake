package globalquake.core.earthquake.data;

import globalquake.core.alert.Warnable;
import globalquake.core.station.AbstractStation;
import globalquake.core.analysis.Event;
import globalquake.core.training.EarthquakeAnalysisTraining;
import globalquake.utils.GeoUtils;

import java.util.List;
import java.awt.*;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;

public class Cluster implements Warnable {

	private final int id;
	private final Map<AbstractStation, Event> assignedEvents;
	private double rootLat;
	private double rootLon;
	private double size;
	public int updateCount;
	private long lastUpdate;

	private Earthquake earthquake;
	@SuppressWarnings("unused")
	public final double bestAngle;
	private Hypocenter previousHypocenter;

	private int level;

	public int lastEpicenterUpdate;

	private double anchorLon;
	private double anchorLat;
	public int revisionID;

	public static final double NONE = -999;

	public final Color color = randomColor();

	private Color randomColor() {
		Random random = new Random();

		// Generate random values for the red, green, and blue components
		int red = random.nextInt(256); // 0-255
		int blue = random.nextInt(256); // 0-255

		// Create a new Color object with the random values
		return new Color(red, 255, blue);
	}


	public Cluster(int id) {
		this.id = id;
		this.assignedEvents = new ConcurrentHashMap<>();
		this.updateCount = 0;
		this.earthquake = null;
		this.bestAngle = -1;
		this.rootLat = NONE;
		this.rootLon = NONE;
		this.lastUpdate = System.currentTimeMillis();
	}

	public Hypocenter getPreviousHypocenter() {
		return previousHypocenter;
	}

	public void setPreviousHypocenter(Hypocenter previousHypocenter) {
		this.previousHypocenter = previousHypocenter;
	}

	public int getId() {
		return id;
	}

	public void addEvent() {
		lastUpdate = System.currentTimeMillis();
	}

	/**
	 * 
	 * @return all events that were added to this cluster
	 */
	public Map<AbstractStation, Event> getAssignedEvents() {
		return assignedEvents;
	}

	public void tick() {
		if (checkForUpdates()) {
			if (rootLat == NONE)
				calculateRoot();
			calculateSize();
			lastUpdate = System.currentTimeMillis();
		}
	}

	

	private boolean checkForUpdates() {
		int upd = 0;
		for (Event e : getAssignedEvents().values()) {
			upd += e.getUpdatesCount();
		}
		boolean b = (upd != updateCount);
		updateCount = upd;
		return b;
	}

	private void calculateSize() {
		double _size = 0;
		int r128 = 0;
		int r1024 = 0;
		int r8192 = 0;
		int r32K = 0;
		for (Event e : getAssignedEvents().values()) {
			if(!e.isValid()){
				continue;
			}
			double dist = GeoUtils.greatCircleDistance(rootLat, rootLon, e.getAnalysis().getStation().getLatitude(),
					e.getAnalysis().getStation().getLongitude());
			if (dist > _size) {
				_size = dist;
			}
			if (e.getMaxRatio() >= 128) {
				r128++;
			}
			if (e.getMaxRatio() >= 1024) {
				r1024++;
			}
			if (e.getMaxRatio() >= 8192) {
				r8192++;
			}
			if (e.getMaxRatio() >= 32768) {
				r32K++;
			}
		}

		int _level = 0;
		if (r128 > 8 || r1024 > 3) {
			_level = 1;
		}
		if (r1024 > 6 || r8192 > 3) {
			_level = 2;
		}
		if (r8192 > 4 || r32K >= 3) {
			_level = 3;
		}
		if (r32K > 3) {
			_level = 4;
		}
		level = _level;
		this.size = _size;
	}

	public void calculateRoot() {
		int n = 0;
		double sumLat = 0;
		double sumLon = 0;
		for (Event e : getAssignedEvents().values()) {
			if(!e.isValid()){
				continue;
			}
			sumLat += e.getLatFromStation();
			sumLon += e.getLonFromStation();
			n++;
		}
		if (n > 0) {
			rootLat = sumLat / n;
			rootLon = sumLon / n;
			anchorLat = rootLat;
			anchorLon = rootLon;
		}
	}

	// For testing only
	public void calculateRoot(List<EarthquakeAnalysisTraining.FakeStation> fakeStations) {
		int n = 0;
		double sumLat = 0;
		double sumLon = 0;
		for (EarthquakeAnalysisTraining.FakeStation fakeStation : fakeStations) {
			sumLat += fakeStation.lat();
			sumLon += fakeStation.lon();
			n++;
		}
		if (n > 0) {
			rootLat = sumLat / n;
			rootLon = sumLon / n;
		}
	}

	public double getRootLat() {
		return rootLat;
	}

	public double getRootLon() {
		return rootLon;
	}

	@SuppressWarnings("unused")
	public double getSize() {
		return size;
	}

	@SuppressWarnings("BooleanMethodIsAlwaysInverted")
	public boolean containsStation(AbstractStation station) {
		return getAssignedEvents().containsKey(station);
	}

	public long getLastUpdate() {
		return lastUpdate;
	}

	public Earthquake getEarthquake() {
		return earthquake;
	}

	public void setEarthquake(Earthquake earthquake) {
		this.earthquake = earthquake;
	}

	public int getActualLevel() {
		return level;
	}

	public void updateAnchor(Hypocenter bestHypocenter) {
		this.anchorLat = bestHypocenter.lat;
		this.anchorLon = bestHypocenter.lon;
	}

	public double getAnchorLat() {
		return anchorLat;
	}

	public double getAnchorLon() {
		return anchorLon;
	}

	@Override
	public String toString() {
		return "Cluster{" +
				"id=" + id +
				", rootLat=" + rootLat +
				", rootLon=" + rootLon +
				", size=" + size +
				", updateCount=" + updateCount +
				", lastUpdate=" + lastUpdate +
				", earthquake=" + earthquake +
				", anchorLon=" + anchorLon +
				", anchorLat=" + anchorLat +
				'}';
	}

	@SuppressWarnings("unused")
	@Override
	public double getWarningLat() {
		return getAnchorLat();
	}

	@SuppressWarnings("unused")
	@Override
	public double getWarningLon() {
		return getAnchorLon();
	}
}
