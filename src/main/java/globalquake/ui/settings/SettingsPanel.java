package globalquake.ui.settings;

import javax.swing.JPanel;

public abstract class SettingsPanel extends JPanel{

	public abstract void save();
	
	public abstract String getTitle();

}
