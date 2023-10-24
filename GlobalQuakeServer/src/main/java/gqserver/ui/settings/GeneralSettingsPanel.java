package gqserver.ui.settings;


import globalquake.core.Settings;
import globalquake.core.geo.DistanceUnit;
import globalquake.core.intensity.IntensityScale;
import globalquake.core.intensity.IntensityScales;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;

public class GeneralSettingsPanel extends SettingsPanel {
	private JComboBox<IntensityScale> comboBoxScale;
	private JCheckBox chkBoxHomeLoc;

	private JTextField textFieldLat;
	private JTextField textFieldLon;
	private JComboBox<DistanceUnit> distanceUnitJComboBox;

	public GeneralSettingsPanel(SettingsFrame settingsFrame) {
		setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));

		createHomeLocationSettings();
		//createAlertsDialogSettings();
		add(createIntensitySettingsPanel());
		createOtherSettings(settingsFrame);

		for(int i = 0; i < 20; i++){
			add(new JPanel()); // fillers
		}
	}

	private void createOtherSettings(SettingsFrame settingsFrame) {
		JPanel panel = new JPanel();
		panel.setBorder(BorderFactory.createTitledBorder("Other"));

		panel.add(new JLabel("Distance units: "));

		distanceUnitJComboBox = new JComboBox<>(DistanceUnit.values());
		distanceUnitJComboBox.setSelectedIndex(Math.max(0, Math.min(distanceUnitJComboBox.getItemCount() - 1, Settings.distanceUnitsIndex)));

		distanceUnitJComboBox.addItemListener(itemEvent -> {
			Settings.distanceUnitsIndex = distanceUnitJComboBox.getSelectedIndex();
			settingsFrame.refreshUI();
        });

		panel.add(distanceUnitJComboBox);

		add(panel);
	}

	private void createHomeLocationSettings() {
		JPanel outsidePanel = new JPanel(new BorderLayout());
		outsidePanel.setBorder(BorderFactory.createTitledBorder("Home location settings"));

		JPanel homeLocationPanel = new JPanel();
		homeLocationPanel.setLayout(new GridLayout(2,1));

		JLabel lblLat = new JLabel("Home Latitude: ");
		JLabel lblLon = new JLabel("Home Longitude: ");

		textFieldLat = new JTextField(20);
		textFieldLat.setText(String.format("%s", Settings.homeLat));
		textFieldLat.setColumns(10);

		textFieldLon = new JTextField(20);
		textFieldLon.setText(String.format("%s", Settings.homeLon));
		textFieldLon.setColumns(10);

		JPanel latPanel = new JPanel();
		//latPanel.setLayout(new BoxLayout(latPanel, BoxLayout.X_AXIS));

		latPanel.add(lblLat);
		latPanel.add(textFieldLat);

		JPanel lonPanel = new JPanel();
		//lonPanel.setLayout(new BoxLayout(lonPanel, BoxLayout.X_AXIS));

		lonPanel.add(lblLon);
		lonPanel.add(textFieldLon);

		homeLocationPanel.add(latPanel);
		homeLocationPanel.add(lonPanel);

		JTextArea infoLocation = new JTextArea("Home location will be used for playing additional alarm \n sounds if an earthquake occurs nearby");
		infoLocation.setBorder(new EmptyBorder(5,5,5,5));
		infoLocation.setLineWrap(true);
		infoLocation.setEditable(false);
		infoLocation.setBackground(homeLocationPanel.getBackground());

		chkBoxHomeLoc = new JCheckBox("Display home location");
		chkBoxHomeLoc.setSelected(Settings.displayHomeLocation);
		outsidePanel.add(chkBoxHomeLoc);

		outsidePanel.add(homeLocationPanel, BorderLayout.NORTH);
		outsidePanel.add(infoLocation, BorderLayout.CENTER);
		outsidePanel.add(chkBoxHomeLoc, BorderLayout.SOUTH);

		add(outsidePanel);
	}

	private JPanel createIntensitySettingsPanel() {
		JPanel panel = new JPanel(new GridLayout(2,1));
		panel.setBorder(BorderFactory.createTitledBorder("Intensity Scale"));

		comboBoxScale = new JComboBox<>(IntensityScales.INTENSITY_SCALES);
		comboBoxScale.setSelectedIndex(Settings.intensityScaleIndex);

		JPanel div = new JPanel();
		div.add(comboBoxScale);
		panel.add(div, BorderLayout.CENTER);

		JLabel lbl = new JLabel();
		lbl.setFont(new Font("Calibri", Font.PLAIN, 13));
		lbl.setText("Keep in mind that the displayed intensities are estimated, not measured");


		panel.add(lbl, BorderLayout.SOUTH);

		return panel;
	}

	@Override
	public void save() {
		Settings.homeLat = parseDouble(textFieldLat.getText(), "Home latitude", -90, 90);
		Settings.homeLon = parseDouble(textFieldLon.getText(), "Home longitude", -180, 180);
		Settings.intensityScaleIndex = comboBoxScale.getSelectedIndex();
		Settings.displayHomeLocation = chkBoxHomeLoc.isSelected();
		Settings.distanceUnitsIndex = distanceUnitJComboBox.getSelectedIndex();
	}

	@Override
	public String getTitle() {
		return "General";
	}
}
