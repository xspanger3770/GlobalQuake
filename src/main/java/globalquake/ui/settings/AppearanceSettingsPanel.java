package globalquake.ui.settings;

import javax.swing.*;

public class AppearanceSettingsPanel extends SettingsPanel{

    private final JCheckBox chkBoxScheme;
    private final JCheckBox chkBoxHomeLoc;

    public AppearanceSettingsPanel() {
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        chkBoxScheme = new JCheckBox("Use old color scheme (exaggerated)");
        chkBoxScheme.setSelected(Settings.useOldColorScheme);
        add(chkBoxScheme);
        chkBoxHomeLoc = new JCheckBox("Display home location");
        chkBoxHomeLoc.setSelected(Settings.displayHomeLocation);
        add(chkBoxHomeLoc);
    }

    @Override
    public void save() {
        Settings.useOldColorScheme = chkBoxScheme.isSelected();
        Settings.displayHomeLocation = chkBoxHomeLoc.isSelected();
    }

    @Override
    public String getTitle() {
        return "Appearance";
    }
}
