package globalquake.ui;

import java.awt.Dimension;
import java.util.ArrayList;

import javax.swing.JFrame;
import javax.swing.JPanel;

import com.morce.globalquake.database.Channel;
import com.morce.globalquake.database.Network;
import com.morce.globalquake.database.Station;

import globalquake.database.SeedlinkManager;

public class StationSelect extends JFrame {

	private final SeedlinkManager seedlinkManager;
	private ArrayList<Station> displayedStations = new ArrayList<>();

	public StationSelect(SeedlinkManager seedlinkManager) {
		this.seedlinkManager = seedlinkManager;
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
		new Thread(() -> {
            ArrayList<Station> list = new ArrayList<>();
			seedlinkManager.getDatabase().getNetworksReadLock().lock();
			try {
				for (Network n : seedlinkManager.getDatabase().getNetworks()) {
					for (Station s : n.getStations()) {
						for (Channel ch : s.getChannels()) {
							if (ch.isAvailable()) {
								list.add(s);
								break;
							}
						}

					}
				}
			}finally {
				seedlinkManager.getDatabase().getNetworksReadLock().unlock();
			}
            System.out.println(list.size() + " Displayed Stations.");
            displayedStations = list;
        }).start();
	}

	public SeedlinkManager getStationManager() {
		return seedlinkManager;
	}
	
	public ArrayList<Station> getDisplayedStations() {
		return displayedStations;
	}

}
