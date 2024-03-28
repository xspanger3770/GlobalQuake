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
import org.tinylog.Logger;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

import static globalquake.core.earthquake.EarthquakeAnalysis.calculateDistances;
import static globalquake.core.earthquake.EarthquakeAnalysis.createListOfExactPickedEvents;

public class GlobalQuakeLab {

    private static final File mainFolder = new File("./training/");
    private static final File archivedFolder = new File(mainFolder,"./training/events/events/M3.00_Poland_2024_03_09_18_56_40/");

    public static void main(String[] args) throws Exception{
        TauPTravelTimeCalculator.init();
        EarthquakeAnalysis.DEPTH_FIX_ALLOWED = false;
        GlobalQuake.prepare(new File(mainFolder, "/settings/"), null);

        Settings.hypocenterDetectionResolution = 80.0;
        Settings.pWaveInaccuracyThreshold = 3000.0;
        Settings.parallelHypocenterLocations = true;

        if(!archivedFolder.exists()){
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
        for(File file : archivedFolder.listFiles()) {
            try {
                tryFile(file);
            }catch(Exception e){
                Logger.error(e);
            }
        }
    }

    private static void tryFile(File file) throws Exception {
        if(file.isDirectory()){
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

        //runTest(archivedQuake);
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

            for (double depth = 0.0; depth <= 600.0; depth += 0.5) {
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
            Logger.debug("Quality: "+cluster.getPreviousHypocenter().quality);
        }
    }

}
