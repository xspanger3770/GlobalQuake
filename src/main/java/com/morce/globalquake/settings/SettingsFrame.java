package com.morce.globalquake.settings;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.EventQueue;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.LinkedList;
import java.util.List;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;

public class SettingsFrame {

	private JFrame frame;
	private JPanel panel;
	private JPanel panel_1;
	private JButton btnSave;
	private JButton btnCancel;

	private List<SettingsPanel> panels = new LinkedList<SettingsPanel>();
	private JTabbedPane tabbedPane;

	public static void main(String[] args) {
		EventQueue.invokeLater(new Runnable() {
			public void run() {
				try {
					SettingsFrame window = new SettingsFrame();
					window.frame.setVisible(true);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
	}

	public SettingsFrame() {
		initialize();
	}

	private void initialize() {
		frame = new JFrame("GlobalQuake Settings");
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		frame.setMinimumSize(new Dimension(400, 300));
		panel = new JPanel(new BorderLayout());
		frame.setContentPane(panel);

		tabbedPane = new JTabbedPane(JTabbedPane.TOP);
		tabbedPane.setPreferredSize(new Dimension(600, 400));
		panel.add(tabbedPane, BorderLayout.CENTER);

		panel_1 = new JPanel();
		panel.add(panel_1, BorderLayout.SOUTH);

		btnCancel = new JButton("Cancel");
		panel_1.add(btnCancel);

		btnCancel.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent e) {
				frame.dispose();
			}
		});

		btnSave = new JButton("Save");
		panel_1.add(btnSave);

		btnSave.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent e) {
				for (SettingsPanel panel : panels) {
					try {
						panel.save();
					} catch (Exception ex) {
						error(ex);
						return;
					}
				}
				Settings.save();
				frame.dispose();
			}
		});

		addPanels();

		frame.pack();
		frame.setLocationRelativeTo(null);
	}

	protected void error(Exception e) {
		JOptionPane.showMessageDialog(frame, e.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
	}

	private void addPanels() {
		panels.add(new GeneralSettingsPanel());
		panels.add(new HypocenterAnalysisPanel());
		panels.add(new AlertsPanel());
		panels.add(new ZejfNetSettingsPanel());

		for (SettingsPanel panel : panels) {
			tabbedPane.addTab(panel.getTitle(), panel);
		}
	}

}
