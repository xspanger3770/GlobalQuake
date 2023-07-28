package globalquake.ui;

import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.Timer;
import java.util.TimerTask;

import javax.swing.JFrame;

import globalquake.core.AbstractStation;

public class StationMonitor extends JFrame {

	public StationMonitor(AbstractStation station) {
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		StationMonitorPanel panel = new StationMonitorPanel(station);
		setContentPane(panel);

		pack();
		setLocationRelativeTo(null);
		setResizable(true);
		setTitle("Station Monitor - " + station.getNetworkCode() + " " + station.getStationCode() + " "
				+ station.getChannelName() + " " + station.getLocationCode());

		Timer timer = new Timer();
		timer.scheduleAtFixedRate(new TimerTask() {
			public void run() {
				panel.updateImage();
				panel.repaint();
			}
		}, 0, 1000);

		addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent e) {
				timer.cancel();
			}
		});

		panel.addComponentListener(new ComponentAdapter() {
			@Override
			public void componentResized(ComponentEvent e) {
				panel.updateImage();
				panel.repaint();
			}
		});
	}

}
