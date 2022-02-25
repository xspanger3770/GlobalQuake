package com.morce.globalquake.ui;

import java.awt.Dimension;
import java.util.ArrayList;

import javax.swing.JFrame;
import javax.swing.JPanel;

import com.morce.globalquake.database.Channel;
import com.morce.globalquake.database.Network;
import com.morce.globalquake.database.Station;
import com.morce.globalquake.database.StationManager;

public class StationSelect extends JFrame {

	private static final long serialVersionUID = 1L;
	private StationManager stationManager;
	private ArrayList<Station> displayedStations = new ArrayList<>();

	public StationSelect(StationManager stationManager) {
		this.stationManager = stationManager;
		loadDisplayed();
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		JPanel panel = new StationSelectPanel(this);
		setPreferredSize(new Dimension(800, 600));
		setContentPane(panel);

		pack();
		setLocationRelativeTo(null);
		setResizable(true);
		setTitle("Select Stations");
	}

	private void loadDisplayed() {
		new Thread() {
			public void run() {
				ArrayList<Station> list = new ArrayList<Station>();
				for (Network n : stationManager.getDatabase().getNetworks()) {
					for (Station s : n.getStations()) {
						for (Channel ch : s.getChannels()) {
							if (ch.isAvailable()) {
								list.add(s);
								break;
							}
						}

					}
				}
				System.out.println(list.size() + " Displayed Stations.");
				displayedStations = list;
			};
		}.start();
	}

	public StationManager getStationManager() {
		return stationManager;
	}
	
	public ArrayList<Station> getDisplayedStations() {
		return displayedStations;
	}

}
