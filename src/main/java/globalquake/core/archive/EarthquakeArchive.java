package globalquake.core.archive;

import globalquake.core.earthquake.ArchivedQuake;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.report.EarthquakeReporter;
import globalquake.main.Main;
import globalquake.ui.settings.Settings;
import globalquake.utils.monitorable.MonitorableCopyOnWriteArrayList;
import org.tinylog.Logger;

import java.io.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class EarthquakeArchive {

	public static final File ARCHIVE_FILE = new File(Main.MAIN_FOLDER, "archive.dat");
	public static final File TEMP_ARCHIVE_FILE = new File(Main.MAIN_FOLDER, "temp_archive.dat");
	private final ExecutorService executor;

	private List<ArchivedQuake> archivedQuakes;

	public EarthquakeArchive() {
		executor = Executors.newSingleThreadExecutor();
	}

	@SuppressWarnings("unchecked")
	public EarthquakeArchive loadArchive() {
		if (!ARCHIVE_FILE.exists()) {
			archivedQuakes = new MonitorableCopyOnWriteArrayList<>();
			Logger.info("Created new archive");
		} else {
			try {
				ObjectInputStream oin = new ObjectInputStream(new FileInputStream(ARCHIVE_FILE));
				archivedQuakes = (MonitorableCopyOnWriteArrayList<ArchivedQuake>) oin.readObject();
				oin.close();
				Logger.info("Loaded " + archivedQuakes.size() + " quakes from archive.");
			} catch (Exception e) {
				Logger.error(e);
				archivedQuakes = new MonitorableCopyOnWriteArrayList<>();
			}
		}

		archivedQuakes.sort(Comparator.comparing(archivedQuake1 -> -archivedQuake1.getOrigin()));

		return this;
	}

	public void saveArchive() {
		if (archivedQuakes != null) {
			try {
				ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(TEMP_ARCHIVE_FILE));
				Logger.info("Saving " + archivedQuakes.size() + " quakes to " + ARCHIVE_FILE.getName());
				out.writeObject(archivedQuakes);
				out.close();
				boolean res = (!ARCHIVE_FILE.exists() || ARCHIVE_FILE.delete()) && TEMP_ARCHIVE_FILE.renameTo(ARCHIVE_FILE);
				if(!res){
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
            archiveQuake(earthquake);

            saveArchive();
            if(Settings.reportsEnabled) {
                reportQuake(earthquake);
            }

        });
	}

	private void reportQuake(Earthquake earthquake) {
		executor.submit(() -> {
            try {
                EarthquakeReporter.report(earthquake);
            } catch (Exception e) {
                Logger.error(e);
            }
        });
	}

	public synchronized void archiveQuake(Earthquake earthquake) {
		if(archivedQuakes == null){
			archivedQuakes = new MonitorableCopyOnWriteArrayList<>();
		}

		ArchivedQuake archivedQuake = new ArchivedQuake(earthquake);
		archivedQuake.updateRegion();
		archivedQuakes.add(0, archivedQuake);
		archivedQuakes.sort(Comparator.comparing(archivedQuake1 -> -archivedQuake1.getOrigin()));

		while(archivedQuakes.size() > Settings.maxArchivedQuakes){
			archivedQuakes.remove(archivedQuakes.size() - 1);
		}
	}

}
