package globalquake.ui.settings;

import javax.swing.*;
import javax.swing.border.EmptyBorder;

public class AppearanceSettingsPanel extends SettingsPanel{

    private final JCheckBox chkBoxScheme;
    private final JCheckBox chkBoxHomeLoc;
    private final JCheckBox chkBoxAntialiasing;

    public AppearanceSettingsPanel() {
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setBorder(new EmptyBorder(5,5,5,5));

        chkBoxScheme = new JCheckBox("Use old color scheme (exaggerated)");
        chkBoxScheme.setSelected(Settings.useOldColorScheme);
        add(chkBoxScheme);

        chkBoxHomeLoc = new JCheckBox("Display home location");
        chkBoxHomeLoc.setSelected(Settings.displayHomeLocation);
        add(chkBoxHomeLoc);

        chkBoxAntialiasing = new JCheckBox("Enable antialiasing for stations");
        chkBoxAntialiasing.setSelected(Settings.antialiasing);
        add(chkBoxAntialiasing);
    }

    @Override
    public void save() {
        Settings.useOldColorScheme = chkBoxScheme.isSelected();
        Settings.displayHomeLocation = chkBoxHomeLoc.isSelected();
        Settings.antialiasing = chkBoxAntialiasing.isSelected();
    }

    @Override
    public String getTitle() {
        return "Appearance";
    }
}
