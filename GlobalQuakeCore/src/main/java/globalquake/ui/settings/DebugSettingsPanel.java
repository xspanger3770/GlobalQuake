package globalquake.ui.settings;

import globalquake.core.Settings;
import globalquake.core.report.EarthquakeReporter;

import javax.swing.*;
import javax.swing.border.EmptyBorder;

public class DebugSettingsPanel extends SettingsPanel {

    private final JCheckBox chkBoxReports;
    private final JCheckBox chkBoxCoreWaves;
    private final JCheckBox chkBoxConfidencePolygons;
    private final JCheckBox chkBoxRevisions;

    public DebugSettingsPanel() {
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setBorder(new EmptyBorder(5, 5, 5, 5));

        add(chkBoxReports = new JCheckBox("Enable Earthquake Reports", Settings.reportsEnabled));
        add(new JLabel("     Reports will be stored in %s".formatted(EarthquakeReporter.ANALYSIS_FOLDER.getPath())));
        add(chkBoxCoreWaves = new JCheckBox("Display PKP and PKIKP Waves", Settings.displayCoreWaves));
        add(chkBoxConfidencePolygons = new JCheckBox("Display epicenter confidence polygons", Settings.confidencePolygons));
        add(chkBoxRevisions = new JCheckBox("Reduce number of revisions", Settings.reduceRevisions));
    }

    @Override
    public void save() {
        Settings.reportsEnabled = chkBoxReports.isSelected();
        Settings.displayCoreWaves = chkBoxCoreWaves.isSelected();
        Settings.confidencePolygons = chkBoxConfidencePolygons.isSelected();
        Settings.reduceRevisions = chkBoxRevisions.isSelected();
    }

    @Override
    public String getTitle() {
        return "Debug";
    }
}
