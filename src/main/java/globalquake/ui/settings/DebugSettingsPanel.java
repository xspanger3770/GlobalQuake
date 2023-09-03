package globalquake.ui.settings;

import globalquake.core.report.EarthquakeReporter;

import javax.swing.*;

public class DebugSettingsPanel extends SettingsPanel {

    private final JCheckBox chkBoxClusters;
    private final JCheckBox chkBoxReports;

    public DebugSettingsPanel() {
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        add(chkBoxClusters = new JCheckBox("Display Clusters", Settings.displayClusters));

        add(chkBoxReports = new JCheckBox("Enable Earthquake Reports", Settings.reportsEnabled));
        add(new JLabel("     Reports will be stored in %s".formatted(EarthquakeReporter.ANALYSIS_FOLDER.getPath())));
    }

    @Override
    public void save() {
        Settings.displayClusters = chkBoxClusters.isSelected();
        Settings.reportsEnabled = chkBoxReports.isSelected();
    }

    @Override
    public String getTitle() {
        return "Debug";
    }
}
