package globalquake.ui.settings;
import javax.swing.JCheckBox;

import globalquake.main.Settings;

public class AlertsPanel extends SettingsPanel {
	private final JCheckBox boxDialogs;

	public AlertsPanel() {
		setLayout(null);
		
		boxDialogs = new JCheckBox("Enable alert dialogs");
		boxDialogs.setBounds(8, 8, 250, 23);
		boxDialogs.setSelected(Settings.enableAlarmDialogs);
		add(boxDialogs);
	}

	@Override
	public void save() {
		Settings.enableAlarmDialogs = boxDialogs.isSelected();
	}

	@Override
	public String getTitle() {
		return "Alerts";
	}
}
