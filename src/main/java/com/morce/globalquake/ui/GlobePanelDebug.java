package com.morce.globalquake.ui;

import java.awt.Dimension;
import java.awt.EventQueue;

import javax.swing.JFrame;
import javax.swing.JPanel;

public class GlobePanelDebug extends JFrame {

	private static final long serialVersionUID = 1L;
	
	public GlobePanelDebug() {
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		JPanel panel = new GlobePanel();
		setPreferredSize(new Dimension(800, 600));
		setContentPane(panel);

		pack();
		setLocationRelativeTo(null);
		setResizable(true);
		setTitle("Globe Panel");
	}

	public static void main(String[] args) {
		EventQueue.invokeLater(new Runnable() {

			@Override
			public void run() {
				new GlobePanelDebug().setVisible(true);
			}
		});
	}
}
