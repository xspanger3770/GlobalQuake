package com.morce.globalquake.ui;

import java.awt.BorderLayout;
import java.awt.Dimension;

import javax.swing.JFrame;
import javax.swing.JPanel;

import com.morce.globalquake.core.GlobalQuake;
import com.morce.globalquake.main.Main;

public class GlobalQuakeFrame extends JFrame {

	private static final long serialVersionUID = 1L;
	private GlobalQuake globalQuake;
	private static final int FPS = 20;


	public GlobalQuakeFrame(GlobalQuake globalQuake) {
		this.globalQuake = globalQuake;
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		GlobalQuakePanel panel = new GlobalQuakePanel(globalQuake,this);
		EarthquakeListPanel list = new EarthquakeListPanel(globalQuake);
		panel.setPreferredSize(new Dimension(600, 600));
		list.setPreferredSize(new Dimension(300, 600));
		
		JPanel mainPanel = new JPanel();
		mainPanel.setLayout(new BorderLayout());
		mainPanel.setPreferredSize(new Dimension(800, 600));
		mainPanel.add(panel,BorderLayout.CENTER);
		mainPanel.add(list,BorderLayout.EAST);
		
		setContentPane(mainPanel);

		pack();
		setLocationRelativeTo(null);
		setMinimumSize(new Dimension(600, 500));
		setResizable(true);
		setTitle(Main.fullName);
		
		new Thread("Main UI Thread") {
			public void run() {
				while (true) {
					try {
						sleep(1000 / FPS);
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
					mainPanel.repaint();
				}
			};
		}.start();
	}

	public GlobalQuake getGlobalQuake() {
		return globalQuake;
	}

}
