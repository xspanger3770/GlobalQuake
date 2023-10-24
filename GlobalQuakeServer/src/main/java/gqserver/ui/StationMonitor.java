package gqserver.ui;

import globalquake.core.station.AbstractStation;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.Timer;
import java.util.TimerTask;

public class StationMonitor extends GQFrame {

	private final AbstractStation station;

	public StationMonitor(Component parent, AbstractStation station, int refreshTime) {
		this.station = station;
		setLayout(new BorderLayout());

		add(createControlPanel(), BorderLayout.NORTH);
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		StationMonitorPanel panel = new StationMonitorPanel(station);
		add(panel, BorderLayout.CENTER);

		pack();

		setLocationRelativeTo(parent);
		setResizable(true);
		setTitle("Station Monitor - " + station.getNetworkCode() + " " + station.getStationCode() + " "
				+ station.getChannelName() + " " + station.getLocationCode());

		Timer timer = new Timer();
		timer.scheduleAtFixedRate(new TimerTask() {
			public void run() {
				panel.updateImage();
				panel.repaint();
			}
		}, 0, refreshTime);

		addWindowListener(new WindowAdapter() {

			@Override
			public void windowClosing(WindowEvent e) {
				timer.cancel();
			}
		});

		addKeyListener(new KeyAdapter() {
			@Override
			public void keyPressed(KeyEvent e) {
				if(e.getKeyCode() == KeyEvent.VK_ESCAPE){
					dispose();
				}
			}
		});

		panel.addComponentListener(new ComponentAdapter() {
			@Override
			public void componentResized(ComponentEvent e) {
				panel.updateImage();
				panel.repaint();
			}
		});

		setVisible(true);
	}

	private Component createControlPanel() {
		JPanel panel = new JPanel();

		JCheckBox chkBoxDisable = new JCheckBox("Disable event picking", station.disabled);
		chkBoxDisable.addActionListener(new AbstractAction() {
			@Override
			public void actionPerformed(ActionEvent actionEvent) {
				station.disabled = chkBoxDisable.isSelected();
			}
		});

		panel.add(chkBoxDisable);

		return panel;
	}

}
