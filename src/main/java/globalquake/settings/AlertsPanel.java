package globalquake.settings;
import javax.swing.JCheckBox;

public class AlertsPanel extends SettingsPanel {
	private JCheckBox boxDialogs;

	public AlertsPanel() {
		setLayout(null);
		
		boxDialogs = new JCheckBox("Enable alert dialogs");
		boxDialogs.setBounds(8, 8, 250, 23);
		boxDialogs.setSelected(Settings.enableAlarmDialogs);
		add(boxDialogs);
	}

	private static final long serialVersionUID = 1L;

	@Override
	public void save() {
		Settings.enableAlarmDialogs = boxDialogs.isSelected();
	}

	@Override
	public String getTitle() {
		return "Alerts";
	}
}
