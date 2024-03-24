package globalquake.ui.settings;

import globalquake.core.Settings;
import globalquake.core.report.EarthquakeReporter;

import javax.swing.*;
import java.awt.*;

public class DebugSettingsPanel extends SettingsPanel {

    private final JCheckBox chkBoxReports;
    private final JCheckBox chkBoxCoreWaves;
    private final JCheckBox chkBoxConfidencePolygons;
    private final JCheckBox chkBoxRevisions;

    public DebugSettingsPanel() {
        setLayout(new BorderLayout());

        JPanel debugPanel = createGridBagPanel();
        debugPanel.add(chkBoxReports = new JCheckBox("Enable Earthquake Reports.", Settings.reportsEnabled), createGbc(0));
        debugPanel.add(createJTextArea("Reports will be stored in %s".formatted(EarthquakeReporter.ANALYSIS_FOLDER.getPath()), this), createGbc(1));
        debugPanel.add(chkBoxCoreWaves = new JCheckBox("Display PKP and PKIKP Waves.", Settings.displayCoreWaves), createGbc(2));
        debugPanel.add(chkBoxConfidencePolygons = new JCheckBox("Display epicenter confidence polygons.", Settings.confidencePolygons), createGbc(3));
        debugPanel.add(chkBoxRevisions = new JCheckBox("Reduce number of revisions", Settings.reduceRevisions), createGbc(4));

        add(debugPanel, BorderLayout.NORTH);
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