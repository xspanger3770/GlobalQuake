package globalquake.main;

import java.io.File;

import globalquake.core.GlobalQuake;
import globalquake.database.StationManager;
import globalquake.ui.DatabaseMonitor;

public class Main {

	private StationManager stationManager;
	private DatabaseMonitor databaseMonitor;
	private GlobalQuake globalQuake;

	public static final String version = "0.8.1";
	public static final String fullName = "GlobalQuake " + version;

	public static final File MAIN_FOLDER = new File("./GlobalQuake/");

	public Main() {
		if (!MAIN_FOLDER.exists()) {
			MAIN_FOLDER.mkdirs();
		}

		startDatabaseManager();
	}

	private void startDatabaseManager() {
		stationManager = new StationManager() {
			@Override
			public void confirmDialog(String title, String message, int optionType, int messageType,
					String... options) {
				super.confirmDialog(title, message, optionType, messageType, options);
				databaseMonitor.confirmDialog(title, message, optionType, messageType, options);
			}
		};

		databaseMonitor = new DatabaseMonitor(stationManager, this);
		databaseMonitor.setVisible(true);
		
		System.out.println("init");
		stationManager.init();
	}

	public static void main(String[] args) {
		new Main();
	}

	public void launch() {
		System.gc();
		System.out.println("Launch");
		globalQuake = new GlobalQuake(stationManager);
	}

	public GlobalQuake getGlobalQuake() {
		return globalQuake;
	}

}
