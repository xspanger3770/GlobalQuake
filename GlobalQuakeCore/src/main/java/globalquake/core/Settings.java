package globalquake.core;

import globalquake.core.exception.RuntimeApplicationException;
import globalquake.core.geo.DistanceUnit;
import globalquake.core.intensity.IntensityScales;
import org.tinylog.Logger;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.time.temporal.TemporalAccessor;
import java.util.Properties;

import java.lang.reflect.Field;
import java.util.function.Function;

public final class Settings {

	private static final File optionsFile = new File(GlobalQuake.mainFolder, "globalQuake.properties");
	private static final Properties properties = new Properties();

	public static Double homeLat;
	public static Double homeLon;

	public static final double pWaveInaccuracyThresholdDefault = 1800;
	public static Double pWaveInaccuracyThreshold;
	public static final double hypocenterCorrectThresholdDefault = 50;
	public static Double hypocenterCorrectThreshold;

	public static final double hypocenterDetectionResolutionDefault = 40;
	public static final double hypocenterDetectionResolutionMax = 160.0;
	public static Double hypocenterDetectionResolution;

	public static Boolean parallelHypocenterLocations;
	public static final int minimumStationsForEEWDefault = 5;

	public static Integer minimumStationsForEEW;

	public static Boolean displayArchivedQuakes;

	public static Boolean useOldColorScheme;

	public static Boolean displayHomeLocation;

	public static Boolean antialiasing;

	public static Integer fpsIdle;

	public static Integer intensityScaleIndex;
	
	public static Boolean reportsEnabled = false;
	public static Boolean enableSound = true;
	public static Boolean oldEventsTimeFilterEnabled;
	public static Double oldEventsTimeFilter;
	public static Boolean oldEventsMagnitudeFilterEnabled;
	public static Double oldEventsMagnitudeFilter;

	public static int changes = 0;

	public static Double oldEventsOpacity;

	public static Boolean displayClusters;
	public static Integer selectedDateFormatIndex;

	public static Integer maxArchivedQuakes;

	public static final DateTimeFormatter[] DATE_FORMATS = {
			DateTimeFormatter.ofPattern("dd/MM/yyyy").withZone(ZoneId.systemDefault()),
			DateTimeFormatter.ofPattern("MM/dd/yyyy").withZone(ZoneId.systemDefault()),
			DateTimeFormatter.ofPattern("yyyy/MM/dd").withZone(ZoneId.systemDefault()),
	};

	public static Boolean use24HFormat;
	public static Double stationIntensityVisibilityZoomLevel;
	public static Boolean hideDeadStations;

	public static final DateTimeFormatter formatter24H = DateTimeFormatter.ofPattern("HH:mm:ss").withZone(ZoneId.systemDefault());
	public static final DateTimeFormatter formatter12H = DateTimeFormatter.ofPattern("hh:mm:ss").withZone(ZoneId.systemDefault());

	public static Boolean alertLocal;
	public static Double alertLocalDist;
	public static Boolean alertRegion;
	public static Double alertRegionMag;
	public static Double alertRegionDist;
	public static Boolean alertGlobal;
	public static Double alertGlobalMag;

	public static Integer cinemaModeSwitchTime;
	public static Integer cinemaModeZoomMultiplier;

	public static Boolean cinemaModeOnStartup;
	public static Boolean cinemaModeReenable;

	public static Integer logsStoreTimeMinutes;
	public static Integer maxEvents;
	public static final int maxEventsDefault = 60;
	public static Boolean displayCoreWaves;
	public static Boolean recalibrateOnLaunch;
	public static Boolean stationsTriangles;
	public static Double stationsSizeMul;
    public static Integer selectedEventColorIndex;

	public static Integer distanceUnitsIndex;
	public static Boolean focusOnEvent;
	public static Boolean jumpToAlert;

	public static Boolean confidencePolygons;

	public static Boolean displayAdditionalQuakeInfo;

	public static Boolean displayMagnitudeHistogram;

	public static Boolean displaySystemInfo;
	public static Boolean reduceRevisions;

	public static Integer shakingLevelScale;
	public static Integer shakingLevelIndex;

	public static Integer strongShakingLevelScale;
	public static Integer strongShakingLevelIndex;

	public static Boolean displayAlertBox;
	public static Boolean displayTime;

	static {
		load();
	}

	private static void load() {
		try {
			properties.load(new FileInputStream(optionsFile));
		} catch (IOException e) {
			Logger.info("Created GlobalQuake properties file at "+optionsFile.getAbsolutePath());
		}

		loadProperty("shakingLevelScale", "0",
				o -> validateInt(0, IntensityScales.INTENSITY_SCALES.length - 1, (Integer) o));
		loadProperty("shakingLevelIndex", "0",
				o -> validateInt(0, IntensityScales.INTENSITY_SCALES[shakingLevelScale].getLevels().size() - 1, (Integer) o));

		loadProperty("strongShakingLevelScale", "0",
				o -> validateInt(0, IntensityScales.INTENSITY_SCALES.length - 1, (Integer) o));
		loadProperty("strongShakingLevelIndex", "5",
				o -> validateInt(0, IntensityScales.INTENSITY_SCALES[strongShakingLevelScale].getLevels().size() - 1, (Integer) o));

		loadProperty("reduceRevisions", "true");

		loadProperty("displayTime", "true");
		loadProperty("displayAlertBox", "true");
		loadProperty("displaySystemInfo", "true");
		loadProperty("displayMagnitudeHistogram", "true");
		loadProperty("displayAdditionalQuakeInfo", "false");

		loadProperty("confidencePolygons", "false");

		loadProperty("focusOnEvent", "true");
		loadProperty("jumpToAlert", "true");

		loadProperty("distanceUnitsIndex", "0", o -> validateInt(0, DistanceUnit.values().length - 1, (Integer) o));

		loadProperty("selectedEventColorIndex", "0", o -> validateInt(0, 2, (Integer) o));
		loadProperty("stationsSizeMul", "1.0",  o -> validateDouble(0, 10, (Double) o));
		loadProperty("recalibrateOnLaunch", "true");

		loadProperty("displayCoreWaves", "false");
		loadProperty("maxEvents", String.valueOf(maxEventsDefault));

		loadProperty("logsStoreTimeMinutes", "5", o -> validateInt(1, 60, (Integer) o));

		loadProperty("cinemaModeOnStartup", "true");
		loadProperty("cinemaModeReenable", "true");
		loadProperty("cinemaModeSwitchTime", "10", o -> validateInt(1, 60 * 60, (Integer) o));
		loadProperty("cinemaModeZoomMultiplier", "100",  o -> validateInt(1, 1000, (Integer) o));

		loadProperty("alertLocal", "true");
		loadProperty("alertLocalDist", "200",  o -> validateDouble(0, 30000, (Double) o));
		loadProperty("alertRegion", "true");
		loadProperty("alertRegionMag", "3.5",  o -> validateDouble(0, 10, (Double) o));
		loadProperty("alertRegionDist", "1000",  o -> validateDouble(0, 30000, (Double) o));
		loadProperty("alertGlobal", "true");
		loadProperty("alertGlobalMag", "6.0",  o -> validateDouble(0, 10, (Double) o));

		loadProperty("reportsEnabled", "false");
		loadProperty("displayClusters", "false");
		loadProperty("selectedDateFormatIndex", "0", o -> validateInt(0, DATE_FORMATS.length - 1, (Integer) o));
		loadProperty("stationIntensityVisibilityZoomLevel", "0.2", o -> validateDouble(0, 10, (Double) o));
		loadProperty("use24HFormat", "true");
		loadProperty("hideDeadStations", "false");
		loadProperty("stationsTriangles", "false");
		loadProperty("maxArchivedQuakes", "100", o -> validateInt(1, Integer.MAX_VALUE, (Integer) o));

		loadProperty("enableAlarmDialogs", "false");
		loadProperty("homeLat", "0.0", o -> validateDouble(-90, 90, (Double) o));
		loadProperty("homeLon", "0.0", o -> validateDouble(-180, 180, (Double) o));
		loadProperty("displayArchivedQuakes", "true");
		loadProperty("enableSound", "true");
		loadProperty("pWaveInaccuracyThreshold", String.valueOf(pWaveInaccuracyThresholdDefault));
		loadProperty("hypocenterCorrectThreshold", String.valueOf(hypocenterCorrectThresholdDefault));
		loadProperty("hypocenterDetectionResolution", String.valueOf(hypocenterDetectionResolutionDefault));
		loadProperty("minimumStationsForEEW", String.valueOf(minimumStationsForEEWDefault));
		loadProperty("useOldColorScheme", "false");
		loadProperty("parallelHypocenterLocations", "true");
		loadProperty("displayHomeLocation", "true");
		loadProperty("antialiasing", "false");
		loadProperty("fpsIdle", "30", o -> validateInt(1, 300, (Integer) o));
		loadProperty("intensityScaleIndex", "0", o -> validateInt(0, IntensityScales.INTENSITY_SCALES.length - 1, (Integer) o));
		loadProperty("oldEventsTimeFilterEnabled", "false");
		loadProperty("oldEventsTimeFilter", "24.0", o -> validateDouble(0, 24 * 365, (Double) o));
		loadProperty("oldEventsMagnitudeFilterEnabled", "false");
		loadProperty("oldEventsMagnitudeFilter", "4.0", o -> validateDouble(0, 10, (Double) o));
		loadProperty("oldEventsOpacity", "100.0", o -> validateDouble(0, 100, (Double) o));

		save();
	}


	public static String formatDateTime(TemporalAccessor temporalAccessor) {
		return selectedDateTimeFormat().format(temporalAccessor) +
				" " +
				(use24HFormat ? formatter24H : formatter12H).format(temporalAccessor);
	}

	private static DateTimeFormatter selectedDateTimeFormat(){
		int i = Math.max(0, Math.min(DATE_FORMATS.length - 1, selectedDateFormatIndex));
		return DATE_FORMATS[i];
	}

	public static DistanceUnit getSelectedDistanceUnit(){
		return DistanceUnit.values()[Math.max(0, Math.min(DistanceUnit.values().length - 1, distanceUnitsIndex))];
	}

	public static boolean validateDouble(double min, double max, double v){
        return !Double.isInfinite(v) && !Double.isNaN(v) && !(v < min) && !(v > max);
    }

	public static boolean validateInt(int min, int max, int v){
		return !(v < min) && !(v > max);
	}

	private static void setProperty(Field field, Object value) {
		try {
			field.set(null, value);
		} catch (Exception e) {
			Logger.error(e);
		}
	}

	private static void loadProperty(String name, String defaultVal){
		loadProperty(name, defaultVal, null);
	}

	private static void loadProperty(String name, String defaultVal, Function<Object, Boolean> validator){
		try {
			Field field = Settings.class.getDeclaredField(name);
			field.setAccessible(true);
			if (field.getType() == Boolean.class) {
				boolean val;
				try{
					val = Boolean.parseBoolean((String) properties.getOrDefault(field.getName(), defaultVal));
					if(validator != null && !validator.apply(val)){
						throw new RuntimeApplicationException("Field %s has invalid value! %s".formatted(name, val));
					}
				}catch(Exception e){
					Logger.error(e);
					val = Boolean.parseBoolean(defaultVal);
				}
				setProperty(field, val);
			} else if (field.getType() == Double.class) {
				double val;
				try{
					val = Double.parseDouble((String) properties.getOrDefault(field.getName(), defaultVal));
					if(validator != null && !validator.apply(val)){
						throw new RuntimeApplicationException("Field %s has invalid value: %s, setting to %s".formatted(name, val, defaultVal));
					}
				}catch(Exception e){
					Logger.error(e);
					val = Double.parseDouble(defaultVal);
				}
				setProperty(field, val);
			} else if (field.getType() == Integer.class) {
				int val;
				try{
					val = Integer.parseInt((String) properties.getOrDefault(field.getName(), defaultVal));
					if(validator != null && !validator.apply(val)){
						throw new RuntimeApplicationException("Field %s has invalid value! %s".formatted(name, val));
					}
				}catch(Exception e){
					Logger.error(e);
					val = Integer.parseInt(defaultVal);
				}
				setProperty(field, val);
			} else {
				Logger.error("Error: unsupported setting type: %s".formatted(field.getType()));
			}
		} catch(NoSuchFieldException ignored){

		} catch (Exception e) {
			Logger.error(e);
		}
	}

	private static Object getPropertyValue(Field field){
		try {
			field.setAccessible(true);
			return field.get(null);
		} catch (Exception e) {
			Logger.error(e);
			return null;
		}
	}
	
	public static void save() {
		changes++;

		try {
			Field[] fields = Settings.class.getDeclaredFields();
			for (Field field : fields) {
				if (field.getType() == Boolean.class || field.getType() == Double.class || field.getType() == Integer.class) {
					String value = String.valueOf(getPropertyValue(field));
					properties.setProperty(field.getName(), value);
				}
			}

			if(!optionsFile.getParentFile().exists()){
				//noinspection ResultOfMethodCallIgnored
				optionsFile.getParentFile().mkdirs();
			}

			properties.store(new FileOutputStream(optionsFile), "Fun fact: I've never felt an earthquake in my life");
		} catch (IOException e) {
			GlobalQuake.getErrorHandler().handleException(e);
		}

	}

}
