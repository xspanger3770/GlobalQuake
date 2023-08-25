package globalquake.ui.settings;

import globalquake.main.Main;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Properties;

public final class Settings {

	public static final double pWaveInaccuracyTresholdDefault = 1200;
	public static final double hypocenterCorrectTresholdDefault = 40;
	private static final File optionsFile = new File(Main.MAIN_FOLDER, "globalQuake.properties");
	private static final Properties properties = new Properties();
	public static Boolean enableAlarmDialogs;
	
	public static Double homeLat;
	public static Double homeLon;

	public static Double pWaveInaccuracyTreshold;
	public static Double hypocenterCorrectTreshold;

	public static Boolean displayArchivedQuakes;
	
	public static final boolean reportsEnabled = false; // not available ATM
	public static Boolean enableSound = true;

	static {
		load();
	}

	private static void load() {
		try {
			properties.load(new FileInputStream(optionsFile));
		} catch (IOException e) {
			System.out.println("Created GlobalQuake properties file at "+optionsFile.getAbsolutePath());
		}
		
		enableAlarmDialogs = Boolean.valueOf((String) properties.getOrDefault("enableAlarmDialogs", "false"));
		
		homeLat = Double.valueOf((String) properties.getOrDefault("homeLat", "0.0"));
		homeLon = Double.valueOf((String) properties.getOrDefault("homeLon", "0.0"));
		displayArchivedQuakes = Boolean.valueOf((String) properties.getOrDefault("displayArchivedQuakes", "true"));
		enableSound = Boolean.valueOf((String) properties.getOrDefault("enableSound", "true"));

		pWaveInaccuracyTreshold = Double.valueOf((String) properties.getOrDefault("pWaveInaccuracyTreshold", String.valueOf(pWaveInaccuracyTresholdDefault)));
		hypocenterCorrectTreshold = Double.valueOf((String) properties.getOrDefault("hypocenterCorrectTreshold", String.valueOf(hypocenterCorrectTresholdDefault)));
		save();
	}
	
	
	public static void save() {
		properties.setProperty("enableAlarmDialogs", String.valueOf(enableAlarmDialogs));
		
		properties.setProperty("homeLat", String.valueOf(homeLat));
		properties.setProperty("homeLon", String.valueOf(homeLon));
		properties.setProperty("displayArchivedQuakes", String.valueOf(displayArchivedQuakes));
		properties.setProperty("enableSound", String.valueOf(enableSound));

		properties.setProperty("pWaveInaccuracyTreshold", String.valueOf(pWaveInaccuracyTreshold));
		properties.setProperty("hypocenterCorrectTreshold", String.valueOf(hypocenterCorrectTreshold));
		try {
			properties.store(new FileOutputStream(optionsFile), "Fun fact: I've never felt an earthquake in my life");
		} catch (IOException e) {
			Main.getErrorHandler().handleException(e);
		}
	}
}
