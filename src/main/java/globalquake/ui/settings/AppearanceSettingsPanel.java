package globalquake.ui.settings;

import javax.swing.*;

public class AppearanceSettingsPanel extends SettingsPanel{

    private final JCheckBox chkBoxScheme;

    public AppearanceSettingsPanel() {
        chkBoxScheme = new JCheckBox("Use old color scheme (exaggerated)");
        chkBoxScheme.setSelected(Settings.useOldColorScheme);
        add(chkBoxScheme);
    }

    @Override
    public void save() {
        Settings.useOldColorScheme = chkBoxScheme.isSelected();
    }

    @Override
    public String getTitle() {
        return "Appearance";
    }
}
