package globalquake.main;

import java.io.File;
import java.io.IOException;

import globalquake.database.StationDatabaseManager;
import globalquake.database_old.SeedlinkManager;
import globalquake.exception.ApplicationErrorHandler;
import globalquake.exception.FatalIOException;
import globalquake.regions.Regions;
import globalquake.sounds.Sounds;
import globalquake.ui.DatabaseMonitorFrameOld;
import globalquake.ui.database.DatabaseMonitorFrame;
import globalquake.utils.Scale;

public class Main {

	private static ApplicationErrorHandler errorHandler;

	public static final String version = "0.9.0";
	public static final String fullName = "GlobalQuake " + version;

	public static final File MAIN_FOLDER = new File("./GlobalQuake/");

	private static void startDatabaseManager() throws IOException {
		StationDatabaseManager manager = new StationDatabaseManager();
		try {
			manager.load();
		} catch (FatalIOException e) {
			getErrorHandler().handleException(e);
		}

		new DatabaseMonitorFrame(manager).setVisible(true);
	}

	public static void main(String[] args) {
		initErrorHandler();

		try {
			Regions.init();
			Scale.load();
			Sounds.load();
		} catch (Exception e) {
			getErrorHandler().handleException(e);
		}

		if (!MAIN_FOLDER.exists()) {
			if(!MAIN_FOLDER.mkdirs()){
				errorHandler.handleException(new FatalIOException("Unable to create main directory!", null));
			}
		}

		try {
			startDatabaseManager();
		} catch (IOException e) {
			errorHandler.handleException(e);
		}
	}

	public static void launchGlobalQuake() {

	}

	public static ApplicationErrorHandler getErrorHandler() {
		return errorHandler;
	}

    public static void initErrorHandler() {
		Thread.setDefaultUncaughtExceptionHandler(errorHandler = new ApplicationErrorHandler(null));
    }
}
