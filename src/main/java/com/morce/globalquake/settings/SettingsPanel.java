package com.morce.globalquake.settings;

import javax.swing.JPanel;

public abstract class SettingsPanel extends JPanel{
	
	private static final long serialVersionUID = 1L;

	public abstract void save();
	
	public abstract String getTitle();

}
