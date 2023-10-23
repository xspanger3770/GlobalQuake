package globalquake.main;

import globalquake.core.GlobalQuake;
import globalquake.database.StationDatabaseManager;
import globalquake.database.StationSource;
import globalquake.exception.ApplicationErrorHandler;
import globalquake.exception.RuntimeApplicationException;
import globalquake.exception.FatalIOException;
import globalquake.geo.taup.TauPTravelTimeCalculator;
import globalquake.intensity.IntensityTable;
import globalquake.intensity.ShakeMap;
import globalquake.regions.Regions;
import globalquake.sounds.Sounds;
import globalquake.training.EarthquakeAnalysisTraining;
import globalquake.ui.client.MainFrame;
import globalquake.ui.database.DatabaseMonitorFrame;
import globalquake.ui.settings.Settings;
import globalquake.utils.Scale;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.util.Objects;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

public class Main {

    private static ApplicationErrorHandler errorHandler;
    public static final String version = "0.10.0_pre1";
    public static final String fullName = "GlobalQuake " + version;
    public static final File MAIN_FOLDER = new File("./GlobalQuake/");
    private static MainFrame mainFrame;

    public static final Image LOGO = new ImageIcon(Objects.requireNonNull(ClassLoader.getSystemClassLoader().getResource("logo/logo.png"))).getImage();

    public static void main(String[] args) {
        initErrorHandler();

        initMainDirectory();

        mainFrame = new MainFrame();
        mainFrame.setVisible(true);
    }

    private static void initMainDirectory() {
        if (!MAIN_FOLDER.exists()) {
            if (!MAIN_FOLDER.mkdirs()) {
                getErrorHandler().handleException(new FatalIOException("Unable to create main directory!", null));
            }
        }
    }

    public static ApplicationErrorHandler getErrorHandler() {
        if(errorHandler == null) {
            errorHandler = new ApplicationErrorHandler(null);
        }
        return errorHandler;
    }

    public static void initErrorHandler() {
        Thread.setDefaultUncaughtExceptionHandler(getErrorHandler());
    }
}
