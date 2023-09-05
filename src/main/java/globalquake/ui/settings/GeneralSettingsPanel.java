package globalquake.ui.settings;

import globalquake.intensity.IntensityScale;
import globalquake.intensity.IntensityScales;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;

public class GeneralSettingsPanel extends SettingsPanel {
	private final JCheckBox chkBoxAlertDialogs;
	private JComboBox<IntensityScale> comboBoxScale;
	private final JCheckBox chkBoxHomeLoc;


	public GeneralSettingsPanel() {
		setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));

		JPanel outsidePanel = new JPanel(new BorderLayout());
		outsidePanel.setBorder(BorderFactory.createTitledBorder("Home location settings"));

		JPanel homeLocationPanel = new JPanel();
		homeLocationPanel.setLayout(new GridLayout(2,2));
		
		JLabel lblLat = new JLabel("Home Latitude: ");
		JLabel lblLon = new JLabel("Home Longitude: ");

		textFieldLat = new JTextField();
		textFieldLat.setText("%s".formatted(Settings.homeLat));
		textFieldLat.setColumns(10);
		
		textFieldLon = new JTextField();
		textFieldLon.setText("%s".formatted(Settings.homeLon));
		textFieldLon.setColumns(10);

		homeLocationPanel.add(lblLat);
		homeLocationPanel.add(textFieldLat);
		homeLocationPanel.add(lblLon);
		homeLocationPanel.add(textFieldLon);

		JTextArea infoLocation = new JTextArea("Home location will be used for playing additional alarm \n sounds if an earthquake occurs nearby");
		infoLocation.setBorder(new EmptyBorder(5,5,5,5));
		infoLocation.setLineWrap(true);
		infoLocation.setEditable(false);
		infoLocation.setBackground(homeLocationPanel.getBackground());

		chkBoxHomeLoc = new JCheckBox("Display home location");
		chkBoxHomeLoc.setSelected(Settings.displayHomeLocation);
		add(chkBoxHomeLoc);

		outsidePanel.add(homeLocationPanel, BorderLayout.NORTH);
		outsidePanel.add(infoLocation, BorderLayout.CENTER);
		outsidePanel.add(chkBoxHomeLoc, BorderLayout.SOUTH);

		add(outsidePanel);

		JPanel alertsDialogPanel = new JPanel(new GridLayout(2, 1));
		alertsDialogPanel.setBorder(BorderFactory.createTitledBorder("Alert dialogs settings"));

		chkBoxAlertDialogs = new JCheckBox("Enable alert dialogs");
		chkBoxAlertDialogs.setBounds(8, 8, 250, 23);
		chkBoxAlertDialogs.setSelected(Settings.enableAlarmDialogs);

		JTextArea textAreaDialogs = new JTextArea(
                """
                        Alert dialog will show if an earthquake occurs\s
                         nearby your home location and will display P and S wave\s
                         arrival time and estimated intensity""");

		textAreaDialogs.setBorder(new EmptyBorder(0,5,5,5));
		textAreaDialogs.setLineWrap(true);
		textAreaDialogs.setEditable(false);
		textAreaDialogs.setBackground(homeLocationPanel.getBackground());

		alertsDialogPanel.add(chkBoxAlertDialogs);
		alertsDialogPanel.add(textAreaDialogs);

		add(alertsDialogPanel);
		add(createIntensitySettingsPanel());

		for(int i = 0; i < 16; i++){
			add(new JPanel()); // fillers
		}
	}

	private JPanel createIntensitySettingsPanel() {
		JPanel panel = new JPanel(new GridLayout(2,1));
		panel.setBorder(BorderFactory.createTitledBorder("Intensity Scale"));

		comboBoxScale = new JComboBox<>(IntensityScales.INTENSITY_SCALES.toArray(new IntensityScale[0]));
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

	private final JTextField textFieldLat;
	private final JTextField textFieldLon;

	@Override
	public void save() {
		Settings.homeLat = Double.valueOf(textFieldLat.getText());
		Settings.homeLon = Double.valueOf(textFieldLon.getText());
		Settings.enableAlarmDialogs = chkBoxAlertDialogs.isSelected();
		Settings.intensityScaleIndex = comboBoxScale.getSelectedIndex();
		Settings.displayHomeLocation = chkBoxHomeLoc.isSelected();
	}

	@Override
	public String getTitle() {
		return "General";
	}
}
