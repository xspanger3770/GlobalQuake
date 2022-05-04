package com.morce.globalquake.core;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;

import com.morce.globalquake.core.report.EarthquakeReporter;
import com.morce.globalquake.core.simulation.FakeGlobalQuake;
import com.morce.globalquake.main.Main;

public class EarthquakeArchive {

	private GlobalQuake globalQuake;
	public static final File ARCHIVE_FILE = new File(Main.MAIN_FOLDER, "archive.dat");
	public static final File TEMP_ARCHIVE_FILE = new File(Main.MAIN_FOLDER, "temp_archive.dat");

	public Object archivedQuakesSync;
	private ArrayList<ArchivedQuake> archivedQuakes;

	public EarthquakeArchive(GlobalQuake globalQuake) {
		this.globalQuake = globalQuake;
		this.archivedQuakesSync = new Object();
		loadArchive();
	}

	@SuppressWarnings("unchecked")
	private void loadArchive() {
		if (!ARCHIVE_FILE.exists()) {
			archivedQuakes = new ArrayList<ArchivedQuake>();
			System.out.println("Created new archive");
		} else {
			try {
				ObjectInputStream oin = new ObjectInputStream(new FileInputStream(ARCHIVE_FILE));
				archivedQuakes = (ArrayList<ArchivedQuake>) oin.readObject();
				oin.close();
				System.out.println("Loaded " + archivedQuakes.size() + " quakes from archive.");
			} catch (Exception e) {
				System.err.println("Failed to load archive");
				e.printStackTrace();
				archivedQuakes = new ArrayList<ArchivedQuake>();
			}
		}
		saveArchive();
		if (getGlobalQuake() instanceof FakeGlobalQuake) {
			ArchivedQuake q;
			// will not be saved
			archivedQuakes.add(q = new ArchivedQuake(69.2, 44.24, 10, 5.0, System.currentTimeMillis(), "debug-quake"));
			for (double ang = 0; ang < 360; ang += 60) {
				q.getArchivedEvents().add(new ArchivedEvent(50 + Math.sin(Math.toRadians(ang)),
						17 + Math.cos(Math.toRadians(ang)), 0, 0, ang % 120 == 0));
			}
		}
	}

	public void saveArchive() {
		if (archivedQuakes != null) {
			try {
				synchronized (archivedQuakesSync) {
					ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(TEMP_ARCHIVE_FILE));
					System.out.println("Saving " + archivedQuakes.size() + " quakes to " + ARCHIVE_FILE.getName());
					out.writeObject(archivedQuakes);
					out.close();
					ARCHIVE_FILE.delete();
					TEMP_ARCHIVE_FILE.renameTo(ARCHIVE_FILE);
					TEMP_ARCHIVE_FILE.delete();
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	public GlobalQuake getGlobalQuake() {
		return globalQuake;
	}

	public ArrayList<ArchivedQuake> getArchivedQuakes() {
		return archivedQuakes;
	}

	public void archiveQuakeAndSave(Earthquake earthquake) {
		new Thread("Archive Thread") {
			public void run() {
				ArchivedQuake archivedQuake = new ArchivedQuake(earthquake);
				synchronized (archivedQuakesSync) {
					archivedQuakes.add(archivedQuake);
					Collections.sort(archivedQuakes, Comparator.comparing(ArchivedQuake::getOrigin));
				}
				saveArchive();
				reportQuake(earthquake);

			};
		}.start();
	}

	private void reportQuake(Earthquake earthquake) {
		new Thread("Quake Reporter") {
			@Override
			public void run() {
				try {
					EarthquakeReporter.report(earthquake);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		}.start();
	}

	public void archiveQuake(Earthquake earthquake) {
		ArchivedQuake archivedQuake = new ArchivedQuake(earthquake);
		synchronized (archivedQuakesSync) {
			archivedQuakes.add(archivedQuake);
			Collections.sort(archivedQuakes, Comparator.comparing(ArchivedQuake::getOrigin));
		}
	}

	public void update() {
		synchronized (archivedQuakesSync) {
			Iterator<ArchivedQuake> it = archivedQuakes.iterator();
			boolean save = false;
			while (it.hasNext()) {
				ArchivedQuake archivedQuake = it.next();
				long age = System.currentTimeMillis() - archivedQuake.getOrigin();
				if (age > 1000l * 60 * 60 * 24 * 3l) {
					if (!save) {
						save = true;
					}
					it.remove();
				}
			}

			if (save) {
				new Thread("Archive Save Thread") {
					public void run() {
						saveArchive();
					};
				}.start();
			}
		}
	}

}
