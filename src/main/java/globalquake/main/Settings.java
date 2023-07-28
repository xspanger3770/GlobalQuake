package globalquake.main;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Properties;

public final class Settings {

	private static final File optionsFile = new File(Main.MAIN_FOLDER, "globalQuake.properties");
	private static final Properties properties = new Properties();
	
	public static String zejfSeisIP;
	public static Integer zejfSeisPort;
	public static Boolean zejfSeisAutoReconnect;
	
	public static Boolean enableAlarmDialogs;
	
	public static Double homeLat;
	public static Double homeLon;
	
	public static boolean reportsEnabled = false; // not available
	public static final boolean enableSound = true; // not available ATM

	static {
		load();
	}

	private static void load() {
		try {
			properties.load(new FileInputStream(optionsFile));
		} catch (IOException e) {
			System.out.println("Created GlobalQuake properties file at "+optionsFile.getAbsolutePath());
		}
		zejfSeisIP = (String) properties.getOrDefault("zejfSeisIP", "0.0.0.0");
		zejfSeisPort = Integer.valueOf((String) properties.getOrDefault("zejfSeisPort", "6222"));
		zejfSeisAutoReconnect = Boolean.valueOf((String) properties.getOrDefault("zejfSeisAutoReconnect", "true"));
		
		enableAlarmDialogs = Boolean.valueOf((String) properties.getOrDefault("enableAlarmDialogs", "false"));
		
		homeLat = Double.valueOf((String) properties.getOrDefault("homeLat", "0.0"));
		homeLon = Double.valueOf((String) properties.getOrDefault("homeLon", "0.0"));
		save();
	}
	
	
	public static void save() {
		properties.setProperty("zejfSeisIP", zejfSeisIP);
		properties.setProperty("zejfSeisPort", String.valueOf(zejfSeisPort));
		properties.setProperty("zejfSeisAutoReconnect", String.valueOf(zejfSeisAutoReconnect));
		properties.setProperty("enableAlarmDialogs", String.valueOf(enableAlarmDialogs));
		
		properties.setProperty("homeLat", String.valueOf(homeLat));
		properties.setProperty("homeLon", String.valueOf(homeLon));
		try {
			properties.store(new FileOutputStream(optionsFile), "magic");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
