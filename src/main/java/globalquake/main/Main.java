package globalquake.main;

import java.io.File;
import java.io.IOException;

import globalquake.database.SeedlinkManager;
import globalquake.exception.ApplicationErrorHandler;
import globalquake.exception.FatalIOException;
import globalquake.regions.Regions;
import globalquake.sounds.Sounds;
import globalquake.ui.DatabaseMonitorFrame;
import globalquake.utils.Scale;

public class Main {

	private static ApplicationErrorHandler errorHandler;
	private static SeedlinkManager seedlinkManager;

	public static final String version = "0.8.1";
	public static final String fullName = "GlobalQuake " + version;

	public static final File MAIN_FOLDER = new File("./GlobalQuake/");

	private static void startDatabaseManager() throws IOException {
		seedlinkManager = new SeedlinkManager() {
		};

		DatabaseMonitorFrame databaseMonitorFrame = new DatabaseMonitorFrame(seedlinkManager) {
			@Override
			public void launch() {
				launchGlobalQuake();
			}
		};
		errorHandler.setParent(databaseMonitorFrame);

		databaseMonitorFrame.setVisible(true);
		seedlinkManager.load();
	}

	public static void main(String[] args) {
		Thread.setDefaultUncaughtExceptionHandler(errorHandler = new ApplicationErrorHandler(null));

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
		GlobalQuake globalQuake = new GlobalQuake(seedlinkManager);
	}

	public static ApplicationErrorHandler getErrorHandler() {
		return errorHandler;
	}
}
