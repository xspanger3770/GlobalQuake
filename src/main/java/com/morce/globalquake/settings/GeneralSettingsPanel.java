package com.morce.globalquake.settings;
import javax.swing.JLabel;
import javax.swing.JTextField;

public class GeneralSettingsPanel extends SettingsPanel {
	public GeneralSettingsPanel() {
		setLayout(null);
		
		JLabel lblLat = new JLabel("Home Lat");
		lblLat.setBounds(12, 12, 70, 15);
		add(lblLat);
		
		JLabel lblLon = new JLabel("Home Lon");
		lblLon.setBounds(12, 39, 70, 15);
		add(lblLon);
		
		textFieldLat = new JTextField();
		textFieldLat.setBounds(100, 10, 114, 19);
		textFieldLat.setText(Settings.homeLat+"");
		add(textFieldLat);
		textFieldLat.setColumns(10);
		
		textFieldLon = new JTextField();
		textFieldLon.setBounds(100, 37, 114, 19);
		textFieldLon.setText(Settings.homeLon+"");
		add(textFieldLon);
		textFieldLon.setColumns(10);
	}

	private static final long serialVersionUID = 1L;
	private JTextField textFieldLat;
	private JTextField textFieldLon;

	@Override
	public void save() {
		Settings.homeLat = Double.valueOf(textFieldLat.getText());
		Settings.homeLon = Double.valueOf(textFieldLon.getText());
	}

	@Override
	public String getTitle() {
		return "General";
	}
}
