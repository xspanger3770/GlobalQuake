package globalquake.core.lab;

import globalquake.core.GlobalQuake;
import globalquake.core.HypocsSettings;
import globalquake.core.Settings;
import globalquake.core.archive.ArchivedEvent;
import globalquake.core.archive.ArchivedQuake;
import globalquake.core.earthquake.EarthquakeAnalysis;
import globalquake.core.earthquake.data.*;
import globalquake.core.geo.taup.TauPTravelTimeCalculator;
import globalquake.core.training.EarthquakeAnalysisTraining;
import globalquake.utils.GeoUtils;
import org.tinylog.Logger;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import static globalquake.core.earthquake.EarthquakeAnalysis.calculateDistances;
import static globalquake.core.earthquake.EarthquakeAnalysis.createListOfExactPickedEvents;

@SuppressWarnings("unused")
public class GlobalQuakeLab {

    private static final File mainFolder = new File("./training/");
    private static final File archivedFolder = new File("/home/xspanger/Desktop/GlobalQuake/training/");
    //private static final File archivedFolder = new File(mainFolder,"./training/events/events/M3.51_east_of_the_North_Island_of_New_Zealand_2024_03_13_12_59_04/");

    public static void main(String[] args) throws Exception {
        TauPTravelTimeCalculator.init();
        EarthquakeAnalysis.DEPTH_FIX_ALLOWED = false;
        GlobalQuake.prepare(new File(mainFolder, "/settings/"), null);

        Settings.hypocenterDetectionResolution = 80.0;
        Settings.pWaveInaccuracyThreshold = 3000.0;
        Settings.parallelHypocenterLocations = true;

        if (!archivedFolder.exists()) {
            //noinspection ResultOfMethodCallIgnored
            archivedFolder.mkdirs();
            System.out.printf("Created archived quakes folder at %s".formatted(archivedFolder.getAbsolutePath()));
        }

        traverseDirectories(archivedFolder);

        HypocsSettings.save();

        System.exit(0);
    }

    @SuppressWarnings("DataFlowIssue")
    public static void traverseDirectories(File folder) {
        if (folder.isDirectory()) {
            Arrays.asList(folder.listFiles()).parallelStream().forEach(file -> {
                try {
                    if (file.isDirectory()) {
                        System.err.println(file.getAbsolutePath());
                        traverseDirectories(file); // Recursive call for subdirectory
                    } else if (file.getName().endsWith(".dat")) {
                        tryFile(file, folder); // Call tryFile for each file
                    }
                } catch (Exception e) {
                    Logger.error(e);
                }
            });
        }
    }

    private static void tryFile(File file, File folder) throws Exception {
        if (file.isDirectory()) {
            return;
        }
        ObjectInputStream in = new ObjectInputStream(new FileInputStream(file));
        ArchivedQuake archivedQuake = (ArchivedQuake) in.readObject();
        inspectArchivedQuake(archivedQuake, folder);
    }

    private static void inspectArchivedQuake(ArchivedQuake archivedQuake, File folder) {
        //System.out.println(archivedQuake);

        depthInspection(archivedQuake, folder);
        try {
            residualsInspection(archivedQuake, folder);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        //runTest(archivedQuake);
    }

    private static void residualsInspection(ArchivedQuake archivedQuake, File folder) throws IOException {
        long origin = archivedQuake.getOrigin();
        String filePath = new File(folder, "residuals.csv").getAbsolutePath();

        double depth = archivedQuake.getDepth();

        try (FileWriter writer = new FileWriter(filePath)) {
            writer.write("angle,azimuth,residual\n");

            archivedQuake.getArchivedEvents().sort(Comparator.comparing(archivedEvent -> GeoUtils.greatCircleDistance(archivedEvent.lat(), archivedEvent.lon(), archivedQuake.getLat(), archivedQuake.getLon())));

            for (ArchivedEvent archivedEvent : archivedQuake.getArchivedEvents()) {
                long arrival = archivedEvent.pWave();
                double angle = TauPTravelTimeCalculator.toAngle(
                        GeoUtils.greatCircleDistance(archivedEvent.lat(), archivedEvent.lon(), archivedQuake.getLat(), archivedQuake.getLon()));
                double travelTime = TauPTravelTimeCalculator.getPWaveTravelTimeFast(depth, angle);
                long expectedArrival = (origin + (long) (travelTime * 1000L));
                double residual = (arrival - expectedArrival) / 1000.0;
                double azimuth = GeoUtils.calculateAngle(archivedEvent.lat(), archivedEvent.lon(), archivedQuake.getLat(), archivedQuake.getLon());
                if (Math.abs(residual) < 15.0 && angle < 20)
                    writer.write("%s,%s,%s\n".formatted(angle, azimuth, residual));
                // System.err.printf("Residual = %.2fs%n", residual);
            }
        }
    }

    private static void depthInspection(ArchivedQuake archivedQuake, File folder) {
        List<PickedEvent> pickedEvents = new ArrayList<>();
        var cluster = new Cluster();
        cluster.updateCount = 6543541;

        List<EarthquakeAnalysisTraining.FakeStation> fakeStations = new ArrayList<>();
        for (ArchivedEvent archivedEvent : archivedQuake.getArchivedEvents()) {
            fakeStations.add(new EarthquakeAnalysisTraining.FakeStation(archivedEvent.lat(), archivedEvent.lon()));
        }

        for (ArchivedEvent archivedEvent : archivedQuake.getArchivedEvents()) {
            var event = new PickedEvent(archivedEvent.pWave(), archivedEvent.lat(), archivedEvent.lon(), 0, archivedEvent.maxRatio());
            pickedEvents.add(event);
        }

        cluster.calculateRoot(fakeStations);
        //System.err.println(cluster);

        HypocenterFinderThreadData threadData = new HypocenterFinderThreadData(fakeStations.size());
        HypocenterFinderSettings finderSettings = EarthquakeAnalysis.createSettings(false);

        List<EarthquakeAnalysis.ExactPickedEvent> exactPickedEvents = createListOfExactPickedEvents(pickedEvents);
        calculateDistances(exactPickedEvents, archivedQuake.getLat(), archivedQuake.getLon());

        String filePath = new File(folder, "heuristic.csv").getAbsolutePath();

        try (FileWriter writer = new FileWriter(filePath)) {
            writer.write("depth,err,correct,heuristic\n");

            for (double depth = 0.0; depth <= 749.0; depth += 0.25) {
                // double depth = 50.0;
                EarthquakeAnalysis.analyseHypocenter(threadData.hypocenterA, archivedQuake.getLat(), archivedQuake.getLon(), depth, exactPickedEvents, finderSettings, threadData);

                double heuristics = EarthquakeAnalysis.calculateHeuristic(threadData.hypocenterA);

                writer.write("%s,%s,%s,%s\n".formatted(depth, threadData.hypocenterA.err, threadData.hypocenterA.correctStations, heuristics));
                //System.err.printf("%s,%s,%s,%s%n", depth, threadData.hypocenterA.err, threadData.hypocenterA.correctStations, heuristics);
            }
        } catch (IOException e) {
            Logger.error(e);
        }
    }

    private static void runTest(ArchivedQuake archivedQuake) {
        EarthquakeAnalysis earthquakeAnalysis = new EarthquakeAnalysis();
        earthquakeAnalysis.testing = true;

        List<EarthquakeAnalysisTraining.FakeStation> fakeStations = new ArrayList<>();
        for (ArchivedEvent archivedEvent : archivedQuake.getArchivedEvents()) {
            fakeStations.add(new EarthquakeAnalysisTraining.FakeStation(archivedEvent.lat(), archivedEvent.lon()));
        }

        List<PickedEvent> pickedEvents = new ArrayList<>();
        var cluster = new Cluster();
        cluster.updateCount = 6543541;

        Hypocenter absolutetyCorrect = new Hypocenter(archivedQuake.getLat(),
                archivedQuake.getLon(), archivedQuake.getDepth(), archivedQuake.getOrigin(), 0, 0, null, null);

        for (ArchivedEvent archivedEvent : archivedQuake.getArchivedEvents()) {
            var event = new PickedEvent(archivedEvent.pWave(), archivedEvent.lat(), archivedEvent.lon(), 0, archivedEvent.maxRatio());
            pickedEvents.add(event);
        }

        cluster.calculateRoot(fakeStations);
        System.err.println(cluster);

        System.err.printf("process with %d stations and %d events%n", fakeStations.size(), pickedEvents.size());

        earthquakeAnalysis.processCluster(cluster, pickedEvents, true);

        Logger.debug("Previous " + absolutetyCorrect);
        Logger.debug("Got           " + cluster.getPreviousHypocenter());
        if (cluster.getPreviousHypocenter() != null) {
            Logger.debug("Quality: " + cluster.getPreviousHypocenter().quality);
        }
    }

}
