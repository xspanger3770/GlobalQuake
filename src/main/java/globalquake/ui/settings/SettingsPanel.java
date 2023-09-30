package globalquake.ui.settings;

import javax.swing.*;
import java.text.ParseException;

public abstract class SettingsPanel extends JPanel{

	public abstract void save() throws NumberFormatException, ParseException;
	
	public abstract String getTitle();

	public void refreshUI() {}


}
