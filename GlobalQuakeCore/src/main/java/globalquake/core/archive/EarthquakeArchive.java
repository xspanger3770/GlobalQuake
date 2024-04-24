package globalquake.core.archive;

import globalquake.core.GlobalQuake;
import globalquake.core.Settings;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.events.specific.QuakeArchiveEvent;
import globalquake.core.report.EarthquakeReporter;
import globalquake.utils.monitorable.MonitorableCopyOnWriteArrayList;
import org.tinylog.Logger;

import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class EarthquakeArchive {

    public static final File ARCHIVE_FILE = new File(GlobalQuake.mainFolder, "volume/archive.dat");
    public static final File TEMP_ARCHIVE_FILE = new File(GlobalQuake.mainFolder, "volume/temp_archive.dat");
    private final ExecutorService executor;

    private List<ArchivedQuake> archivedQuakes = new MonitorableCopyOnWriteArrayList<>();

    private final Map<UUID, ArchivedQuake> uuidArchivedQuakeMap = new ConcurrentHashMap<>();

    public EarthquakeArchive() {
        executor = Executors.newSingleThreadExecutor();
    }

    @SuppressWarnings("unchecked")
    public EarthquakeArchive loadArchive() {
        if (!ARCHIVE_FILE.exists()) {
            Logger.info("Created new archive");
        } else {
            try {
                ObjectInputStream oin = new ObjectInputStream(new FileInputStream(ARCHIVE_FILE));
                archivedQuakes = (MonitorableCopyOnWriteArrayList<ArchivedQuake>) oin.readObject();
                oin.close();
                Logger.info("Loaded " + archivedQuakes.size() + " quakes from archive.");
            } catch (Exception e) {
                Logger.error(e);
            }
        }

        archivedQuakes.sort(Comparator.comparing(archivedQuake1 -> -archivedQuake1.getOrigin()));
        buildUUIDMap();

        return this;
    }

    private void buildUUIDMap() {
        for (ArchivedQuake archivedQuake : archivedQuakes) {
            uuidArchivedQuakeMap.put(archivedQuake.getUuid(), archivedQuake);
        }
    }

    public void saveArchive() {
        if (archivedQuakes != null) {
            try {
                ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(TEMP_ARCHIVE_FILE));
                Logger.info("Saving " + archivedQuakes.size() + " quakes to " + ARCHIVE_FILE.getName());
                out.writeObject(archivedQuakes);
                out.close();
                boolean res = (!ARCHIVE_FILE.exists() || ARCHIVE_FILE.delete()) && TEMP_ARCHIVE_FILE.renameTo(ARCHIVE_FILE);
                if (!res) {
                    Logger.error("Unable to save archive!");
                } else {
                    Logger.info("Archive saved");
                }
            } catch (Exception e) {
                Logger.error(e);
            }
        }
    }

    public List<ArchivedQuake> getArchivedQuakes() {
        return archivedQuakes;
    }

    public void archiveQuakeAndSave(Earthquake earthquake) {
        executor.submit(() -> {
            try {
                archiveQuake(earthquake);

                saveArchive();
            } catch (Exception e) {
                Logger.error(e);
            }
        });
    }

    private void reportQuake(Earthquake earthquake, ArchivedQuake archivedQuake) {
        executor.submit(() -> {
            try {
                EarthquakeReporter.report(earthquake, archivedQuake);
            } catch (Exception e) {
                Logger.error(e);
            }
        });
    }

    public void archiveQuake(Earthquake earthquake) {
        ArchivedQuake archivedQuake = new ArchivedQuake(earthquake);
        archiveQuake(archivedQuake, earthquake);
        if (Settings.reportsEnabled) {
            reportQuake(earthquake, archivedQuake);
        }
    }

    protected synchronized void archiveQuake(ArchivedQuake archivedQuake, Earthquake earthquake) {
        if (archivedQuakes == null) {
            archivedQuakes = new MonitorableCopyOnWriteArrayList<>();
        }

        archivedQuake.updateRegion();
        archivedQuakes.add(0, archivedQuake);
        uuidArchivedQuakeMap.put(archivedQuake.getUuid(), archivedQuake);
        archivedQuakes.sort(Comparator.comparing(archivedQuake1 -> -archivedQuake1.getOrigin()));

        if (GlobalQuake.instance != null && earthquake != null) {
            GlobalQuake.instance.getEventHandler().fireEvent(new QuakeArchiveEvent(earthquake, archivedQuake));
        }

        while (archivedQuakes.size() > Settings.maxArchivedQuakes) {
            ArchivedQuake toRemove = archivedQuakes.get(archivedQuakes.size() - 1);
            archivedQuakes.remove(toRemove);
            uuidArchivedQuakeMap.remove(toRemove.getUuid());
        }

        if (archivedQuakes.size() != uuidArchivedQuakeMap.size()) {
            Logger.error("Possible memory leak: %d archived quake, but %d in map".formatted(archivedQuakes.size(), uuidArchivedQuakeMap.size()));
        }
    }

    public ArchivedQuake getArchivedQuakeByUUID(UUID uuid) {
        return uuidArchivedQuakeMap.get(uuid);
    }

    public void destroy() {
        GlobalQuake.instance.stopService(executor);
    }

}
