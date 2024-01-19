package globalquake.core.lab;

import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import globalquake.core.archive.ArchivedEvent;
import globalquake.core.archive.ArchivedQuake;
import globalquake.core.earthquake.EarthquakeAnalysis;
import globalquake.core.earthquake.data.Cluster;
import globalquake.core.earthquake.data.Hypocenter;
import globalquake.core.earthquake.data.PickedEvent;
import globalquake.core.geo.taup.TauPTravelTimeCalculator;
import globalquake.core.training.EarthquakeAnalysisTraining;
import globalquake.utils.GeoUtils;
import org.tinylog.Logger;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class GlobalQuakeLab {

    private static File archivedFolder = new File("./TrainingData/archived/");

    public static void main(String[] args) throws Exception{
        TauPTravelTimeCalculator.init();
        EarthquakeAnalysis.DEPTH_FIX_ALLOWED = false;
        GlobalQuake.prepare(new File(archivedFolder, "/gq/"), null);

        Settings.hypocenterDetectionResolution = 80.0;
        Settings.pWaveInaccuracyThreshold = 4000.0;
        Settings.parallelHypocenterLocations = true;

        if(!archivedFolder.exists()){
            archivedFolder.mkdirs();
            System.out.printf("Created archived quakes folder at %s".formatted(archivedFolder.getAbsolutePath()));
        }

        findFiles();

        System.exit(0);
    }

    private static void findFiles() throws Exception{
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

        runTest(archivedQuake);
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
            System.err.println(archivedEvent);
            pickedEvents.add(event);
        }

        cluster.calculateRoot(fakeStations);
        System.err.println(cluster);

        System.err.println("process with %d stations and %d events".formatted(fakeStations.size(), pickedEvents.size()));

        earthquakeAnalysis.processCluster(cluster, pickedEvents);

        Logger.debug("Previous " + absolutetyCorrect);
        Logger.debug("Got           " + cluster.getPreviousHypocenter());
        if (cluster.getPreviousHypocenter() != null) {
            Logger.debug("Quality: "+cluster.getPreviousHypocenter().quality);
        }
    }

}
