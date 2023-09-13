package globalquake.ui.settings;

import globalquake.core.report.EarthquakeReporter;

import javax.swing.*;
import javax.swing.border.EmptyBorder;

public class DebugSettingsPanel extends SettingsPanel {

    private final JCheckBox chkBoxClusters;
    private final JCheckBox chkBoxReports;
    private final JCheckBox chkBoxCoreWaves;

    public DebugSettingsPanel() {
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setBorder(new EmptyBorder(5,5,5,5));
        add(chkBoxClusters = new JCheckBox("Display Clusters", Settings.displayClusters));

        add(chkBoxReports = new JCheckBox("Enable Earthquake Reports", Settings.reportsEnabled));
        add(new JLabel("     Reports will be stored in %s".formatted(EarthquakeReporter.ANALYSIS_FOLDER.getPath())));
        add(chkBoxCoreWaves = new JCheckBox("Display PKP and PKIKP Waves", Settings.displayCoreWaves));
    }

    @Override
    public void save() {
        Settings.displayClusters = chkBoxClusters.isSelected();
        Settings.reportsEnabled = chkBoxReports.isSelected();
        Settings.displayCoreWaves = chkBoxCoreWaves.isSelected();
    }

    @Override
    public String getTitle() {
        return "Debug";
    }
}
