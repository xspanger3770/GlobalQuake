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
import java.util.Comparator;
import java.util.List;

import static globalquake.core.earthquake.EarthquakeAnalysis.calculateDistances;
import static globalquake.core.earthquake.EarthquakeAnalysis.createListOfExactPickedEvents;

public class GlobalQuakeLab {

    private static final File mainFolder = new File("./training/");
    //private static final File archivedFolder = new File("/home/jakub/Desktop/GlobalQuake/training/events/events/M4.67_Provincia_di_Pordenone,_Italy_2024_03_27_21_19_37/");
    private static final File archivedFolder = new File(mainFolder,"./training/events/events/M3.51_east_of_the_North_Island_of_New_Zealand_2024_03_13_12_59_04/");

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

        findFiles();

        HypocsSettings.save();

        System.exit(0);
    }

    @SuppressWarnings("DataFlowIssue")
    private static void findFiles() {
        for (File file : archivedFolder.listFiles()) {
            try {
                tryFile(file);
            } catch (Exception e) {
                Logger.error(e);
            }
        }
    }

    private static void tryFile(File file) throws Exception {
        if (file.isDirectory()) {
            return;
        }
        ObjectInputStream in = new ObjectInputStream(new FileInputStream(file));
        ArchivedQuake archivedQuake = (ArchivedQuake) in.readObject();
        inspectArchivedQuake(archivedQuake);
    }

    private static void inspectArchivedQuake(ArchivedQuake archivedQuake) {
        System.out.println(archivedQuake);

        /** HOW:
         * [2024-02-22 16:11:49] DEBUG: Hypocenter{totalErr=1.7976931348623157E308, correctEvents=6, lat=-15.049294471740723, lon=-104.17449188232422, depth=716.5, origin=1708617187426, selectedEvents=11, magnitude=5.947031779428777, mags=[MagnitudeReading[magnitude=6.022983368320283, distance=14087.3446279849, eventAge=246800], MagnitudeReading[magnitude=4.798863024224573, distance=14109.64226820795, eventAge=244325], MagnitudeReading[magnitude=5.321131406557377, distance=14110.037034335299, eventAge=244900], MagnitudeReading[magnitude=5.947031779428777, distance=14162.969490055464, eventAge=318400], MagnitudeReading[magnitude=7.033905968753155, distance=15279.572844378148, eventAge=274920], MagnitudeReading[magnitude=8.42564223128777, distance=15989.629923569568, eventAge=263090], MagnitudeReading[magnitude=5.946263594638879, distance=17316.639339086327, eventAge=221420]], obviousArrivalsInfo=ObviousArrivalsInfo{total=6361, wrong=6289}, depthConfidenceInterval=HypocenterConfidenceInterval{minDepth=716.5, maxDepth=716.5}}
         * [2024-02-22 16:11:49] TRACE: Hypocenter finding finished in: 53 ms
         */

        depthInspection(archivedQuake);
        try {
            residualsInspection(archivedQuake);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        //runTest(archivedQuake);
    }

    private static void residualsInspection(ArchivedQuake archivedQuake) throws IOException {
        long origin = archivedQuake.getOrigin();
        String filePath = "residuals.csv";

        double depth = archivedQuake.getDepth();

        try (FileWriter writer = new FileWriter(filePath)) {
            writer.write("angle,azimuth,residual\n");

            archivedQuake.getArchivedEvents().sort(Comparator.comparing(archivedEvent -> GeoUtils.greatCircleDistance(archivedEvent.lat(), archivedEvent.lon(), archivedQuake.getLat(), archivedQuake.getLon())));

            for (ArchivedEvent archivedEvent : archivedQuake.getArchivedEvents()) {
                long arrival = archivedEvent.pWave();
                double angle = TauPTravelTimeCalculator.toAngle(
                        GeoUtils.greatCircleDistance(archivedEvent.lat(), archivedEvent.lon(), archivedQuake.getLat(), archivedQuake.getLon()));
                double travelTime = TauPTravelTimeCalculator.getPWaveTravelTimeFast(depth, angle);
                long expectedArrival = (origin + (long) (travelTime * 1000l));
                double residual = (arrival - expectedArrival) / 1000.0;
                double azimuth = GeoUtils.calculateAngle(archivedEvent.lat(), archivedEvent.lon(), archivedQuake.getLat(), archivedQuake.getLon());
                if (Math.abs(residual) < 15.0 && angle < 20)
                    writer.write("%s,%s,%s\n".formatted(angle, azimuth, residual));
                System.err.println("Residual = %.2fs".formatted(residual));
            }
        }
    }

    private static void depthInspection(ArchivedQuake archivedQuake) {
        List<PickedEvent> pickedEvents = new ArrayList<>();
        var cluster = new Cluster();
        cluster.updateCount = 6543541;

        List<EarthquakeAnalysisTraining.FakeStation> fakeStations = new ArrayList<>();
        for (ArchivedEvent archivedEvent : archivedQuake.getArchivedEvents()) {
            fakeStations.add(new EarthquakeAnalysisTraining.FakeStation(archivedEvent.lat(), archivedEvent.lon()));
        }

        Hypocenter absolutetyCorrect = new Hypocenter(archivedQuake.getLat(),
                archivedQuake.getLon(), archivedQuake.getDepth(), archivedQuake.getOrigin(), 0, 0, null, null);

        for (ArchivedEvent archivedEvent : archivedQuake.getArchivedEvents()) {
            var event = new PickedEvent(archivedEvent.pWave(), archivedEvent.lat(), archivedEvent.lon(), 0, archivedEvent.maxRatio());
            pickedEvents.add(event);
        }

        cluster.calculateRoot(fakeStations);
        System.err.println(cluster);

        HypocenterFinderThreadData threadData = new HypocenterFinderThreadData(fakeStations.size());
        HypocenterFinderSettings finderSettings = EarthquakeAnalysis.createSettings(false);

        List<EarthquakeAnalysis.ExactPickedEvent> exactPickedEvents = createListOfExactPickedEvents(pickedEvents);
        calculateDistances(exactPickedEvents, archivedQuake.getLat(), archivedQuake.getLon());

        String filePath = "output.csv";

        try (FileWriter writer = new FileWriter(filePath)) {
            writer.write("depth,err,correct,heuristic\n");

            for (double depth = 0.0; depth <= 749.0; depth += 0.25) {
                // double depth = 50.0;
                EarthquakeAnalysis.analyseHypocenter(threadData.hypocenterA, archivedQuake.getLat(), archivedQuake.getLon(), depth, exactPickedEvents, finderSettings, threadData);

                double heuristics = EarthquakeAnalysis.calculateHeuristic(threadData.hypocenterA);

                writer.write("%s,%s,%s,%s\n".formatted(depth, threadData.hypocenterA.err, threadData.hypocenterA.correctStations, heuristics));
                System.err.println("%s,%s,%s,%s".formatted(depth, threadData.hypocenterA.err, threadData.hypocenterA.correctStations, heuristics));
            }
        } catch (IOException e) {
            e.printStackTrace();
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
