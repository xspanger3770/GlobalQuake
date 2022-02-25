package com.morce.globalquake.ui;

import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import javax.swing.JFrame;

import com.morce.globalquake.core.GlobalStation;

public class StationMonitor extends JFrame {

	private static final long serialVersionUID = 1L;
	
	public StationMonitor(GlobalStation station) {
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		StationMonitorPanel panel = new StationMonitorPanel(station);
		setContentPane(panel);

		pack();
		setLocationRelativeTo(null);
		setResizable(true);
		setTitle("Station Monitor - " + station.getNetworkCode() + " " + station.getStationCode() + " "
				+ station.getChannelName() + " " + station.getLocationCode());

		Thread uiThread = new Thread("Station Monitor UI Thread") {
			public void run() {
				while (true) {
					try {
						sleep(1000);
					} catch (InterruptedException e) {
						break;
					}
					panel.updateImage();
					panel.repaint();
				}
			};
		};

		uiThread.start();

		addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent e) {
				uiThread.interrupt();
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
