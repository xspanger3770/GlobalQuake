package com.morce.globalquake.main;

import java.awt.EventQueue;
import java.io.File;

import com.morce.globalquake.core.GlobalQuake;
import com.morce.globalquake.database.StationManager;
import com.morce.globalquake.ui.DatabaseMonitor;
import com.morce.globalquake.ui.GlobePanel;

public class Main {

	private StationManager stationManager;
	private DatabaseMonitor databaseMonitor;
	private GlobalQuake globalQuake;

	public static final String version = "0.7.16";
	public static final String fullName = "GlobalQuake " + version;

	public static final File main_folder = new File("./GlobalQuake/");

	public Main() {
		if (!main_folder.exists()) {
			main_folder.mkdirs();
		}

		startDatabaseManager();
	}

	private void startDatabaseManager() {
		new Thread("Init GlobePanel") {
			public void run() {
				GlobePanel.init();
			};
		}.start();
		StationManager.setFolderURL(main_folder.getAbsolutePath() + "/stationDatabase/");
		stationManager = new StationManager() {
			@Override
			public void confirmDialog(String title, String message, int optionType, int messageType,
					String... options) {
				super.confirmDialog(title, message, optionType, messageType, options);
				databaseMonitor.confirmDialog(title, message, optionType, messageType, options);
			}
		};
		stationManager.auto_update = false;

		final Object obj = new Object();

		EventQueue.invokeLater(new Runnable() {

			public void run() {
				databaseMonitor = new DatabaseMonitor(stationManager, Main.this);
				databaseMonitor.setVisible(true);

				synchronized (obj) {
					obj.notify();
				}
			}
		});

		// wait for frame to init
		synchronized (obj) {
			try {
				obj.wait();
			} catch (InterruptedException e1) {
				e1.printStackTrace();
			}
		}
		System.out.println("init");
		stationManager.init();
	}

	public static void main(String[] args) {
		new Main();
	}

	public void launch() {
		System.gc();
		System.out.println("Launch");
		globalQuake = new GlobalQuake(stationManager);
	}

	public GlobalQuake getGlobalQuake() {
		return globalQuake;
	}

}
