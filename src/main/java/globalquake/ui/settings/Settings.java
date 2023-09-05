package globalquake.ui.settings;

import globalquake.main.Main;
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

public final class Settings {

	private static final File optionsFile = new File(Main.MAIN_FOLDER, "globalQuake.properties");
	private static final Properties properties = new Properties();
	public static Boolean enableAlarmDialogs;
	
	public static Double homeLat;
	public static Double homeLon;

	public static final double pWaveInaccuracyThresholdDefault = 1800;
	public static Double pWaveInaccuracyThreshold;
	public static final double hypocenterCorrectThresholdDefault = 50;
	public static Double hypocenterCorrectThreshold;

	public static final double hypocenterDetectionResolutionDefault = 40;
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

	public static String formatDateTime(TemporalAccessor temporalAccessor) {
        return selectedDateTimeFormat().format(temporalAccessor) +
				" " +
				(use24HFormat ? formatter24H : formatter12H).format(temporalAccessor);
	}

	private static DateTimeFormatter selectedDateTimeFormat(){
		int i = Math.max(0, Math.min(DATE_FORMATS.length - 1, selectedDateFormatIndex));
		return DATE_FORMATS[i];
	}

	static {
		load();
	}

	private static void load() {
		try {
			properties.load(new FileInputStream(optionsFile));
		} catch (IOException e) {
			System.out.println("Created GlobalQuake properties file at "+optionsFile.getAbsolutePath());
		}

		loadProperty("cinemaModeSwitchTime", "10");
		loadProperty("cinemaModeZoomMultiplier", "100");

		loadProperty("alertLocal", "true");
		loadProperty("alertLocalDist", "200");
		loadProperty("alertRegion", "true");
		loadProperty("alertRegionMag", "3.5");
		loadProperty("alertRegionDist", "1000");
		loadProperty("alertGlobal", "true");
		loadProperty("alertGlobalMag", "6.0");

		loadProperty("reportsEnabled", "false");
		loadProperty("displayClusters", "false");
		loadProperty("selectedDateFormatIndex", "0");
		loadProperty("stationIntensityVisibilityZoomLevel", "0.2");
		loadProperty("use24HFormat", "true");
		loadProperty("hideDeadStations", "false");
		loadProperty("maxArchivedQuakes", "100");

		loadProperty("enableAlarmDialogs", "false");
		loadProperty("homeLat", "0.0");
		loadProperty("homeLon", "0.0");
		loadProperty("displayArchivedQuakes", "true");
		loadProperty("enableSound", "true");
		loadProperty("pWaveInaccuracyThreshold", String.valueOf(pWaveInaccuracyThresholdDefault));
		loadProperty("hypocenterCorrectThreshold", String.valueOf(hypocenterCorrectThresholdDefault));
		loadProperty("hypocenterDetectionResolution", String.valueOf(hypocenterDetectionResolutionDefault));
		loadProperty("minimumStationsForEEW", String.valueOf(minimumStationsForEEWDefault));
		loadProperty("useOldColorScheme", "false");
		loadProperty("parallelHypocenterLocations", "false");
		loadProperty("displayHomeLocation", "true");
		loadProperty("antialiasing", "false");
		loadProperty("fpsIdle", "30");
		loadProperty("intensityScaleIndex", "0");
		loadProperty("oldEventsTimeFilterEnabled", "false");
		loadProperty("oldEventsTimeFilter", "24.0");
		loadProperty("oldEventsMagnitudeFilterEnabled", "false");
		loadProperty("oldEventsMagnitudeFilter", "4.0");
		loadProperty("oldEventsOpacity", "100.0");

		save();
	}

	private static void setProperty(Field field, Object value) {
		try {
			field.set(null, value);
		} catch (Exception e) {
			Logger.error(e);
		}
	}

	private static void loadProperty(String name, String defaultVal){
		try {
			Field field = Settings.class.getDeclaredField(name);
			field.setAccessible(true);
			if (field.getType() == Boolean.class) {
				boolean val;
				try{
					val = Boolean.parseBoolean((String) properties.getOrDefault(field.getName(), defaultVal));
				}catch(Exception e){
					Logger.error(e);
					val = Boolean.parseBoolean(defaultVal);
				}
				setProperty(field, val);
			} else if (field.getType() == Double.class) {
				double val;
				try{
					val = Double.parseDouble((String) properties.getOrDefault(field.getName(), defaultVal));
				}catch(Exception e){
					Logger.error(e);
					val = Double.parseDouble(defaultVal);
				}
				setProperty(field, val);
			} else if (field.getType() == Integer.class) {
				int val;
				try{
					val = Integer.parseInt((String) properties.getOrDefault(field.getName(), defaultVal));
				}catch(Exception e){
					Logger.error(e);
					val = Integer.parseInt(defaultVal);
				}
				setProperty(field, val);
			}
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
			properties.store(new FileOutputStream(optionsFile), "Fun fact: I've never felt an earthquake in my life");
		} catch (IOException e) {
			Main.getErrorHandler().handleException(e);
		}

	}
}
