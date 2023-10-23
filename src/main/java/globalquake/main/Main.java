package globalquake.main;

import globalquake.exception.ApplicationErrorHandler;
import globalquake.exception.FatalIOException;
import globalquake.ui.client.MainFrame;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.util.Objects;

public class Main {

    private static ApplicationErrorHandler errorHandler;
    public static final String version = "0.10.0_pre1";
    public static final String fullName = "GlobalQuake " + version;
    public static final File MAIN_FOLDER = new File("./GlobalQuake/");

    public static final Image LOGO = new ImageIcon(Objects.requireNonNull(ClassLoader.getSystemClassLoader().getResource("logo/logo.png"))).getImage();

    public static void main(String[] args) {
        initErrorHandler();

        initMainDirectory();

        MainFrame mainFrame = new MainFrame();
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
