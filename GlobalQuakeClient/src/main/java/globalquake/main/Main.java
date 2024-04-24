package globalquake.main;

import globalquake.core.GlobalQuake;
import globalquake.core.earthquake.GQHypocs;
import globalquake.core.exception.ApplicationErrorHandler;
import globalquake.core.exception.FatalIOException;
import globalquake.ui.client.MainFrame;
import org.apache.commons.cli.*;
import org.tinylog.Logger;

import java.io.File;

public class Main {

    private static ApplicationErrorHandler errorHandler;
    public static final String fullName = "GlobalQuake " + GlobalQuake.version;
    public static final File MAIN_FOLDER = new File("./.GlobalQuakeData/");

    public static void main(String[] args) {
        initErrorHandler();
        initMainDirectory();
        GlobalQuake.prepare(MAIN_FOLDER, getErrorHandler());

        Options options = new Options();

        Option maxGpuMemOption = new Option("g", "gpu-max-mem", true, "maximum GPU memory limit in GB");
        maxGpuMemOption.setRequired(false);
        options.addOption(maxGpuMemOption);

        Option performanceTest = new Option("p", "performance-test", false, "run CUDA performance test");
        performanceTest.setRequired(false);
        options.addOption(performanceTest);


        CommandLineParser parser = new org.apache.commons.cli.BasicParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd = null;

        try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            System.err.println(e.getMessage());
            formatter.printHelp("globalquake", options);

            System.exit(1);
        }

        if (cmd.hasOption(maxGpuMemOption.getOpt())) {
            try {
                double maxMem = Double.parseDouble(cmd.getOptionValue(maxGpuMemOption.getOpt()));
                if (maxMem <= 0) {
                    throw new IllegalArgumentException("Invalid maximum GPU memory amount");
                }
                GQHypocs.MAX_GPU_MEM = maxMem;
                Logger.info("Maximum GPU memory allocation will be limited to around %.2f GB".formatted(maxMem));
            } catch (IllegalArgumentException e) {
                Logger.error(e);
                System.exit(1);
            }
        }

        if (cmd.hasOption(performanceTest.getOpt())) {
            try {
                GQHypocs.performanceMeasurement();
            } catch (Exception e) {
                Logger.error(e);
            }
        } else {

            MainFrame mainFrame = new MainFrame();
            mainFrame.setVisible(true);
        }
    }

    private static void initMainDirectory() {
        if (!MAIN_FOLDER.exists()) {
            if (!MAIN_FOLDER.mkdirs()) {
                getErrorHandler().handleException(new FatalIOException("Unable to create main directory!", null));
            }
        }
        File VOLUME_FOLDER = new File(MAIN_FOLDER, "volume/");
        if (!VOLUME_FOLDER.exists()) {
            if (!VOLUME_FOLDER.mkdirs()) {
                getErrorHandler().handleException(new FatalIOException("Unable to create volume directory!", null));
            }
        }
    }

    public static ApplicationErrorHandler getErrorHandler() {
        if (errorHandler == null) {
            errorHandler = new ApplicationErrorHandler(null, false);
        }
        return errorHandler;
    }

    public static void initErrorHandler() {
        Thread.setDefaultUncaughtExceptionHandler(getErrorHandler());
    }
}
