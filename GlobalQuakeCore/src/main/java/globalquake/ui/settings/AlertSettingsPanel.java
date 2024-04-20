package globalquake.ui.settings;

import globalquake.core.Settings;
import globalquake.core.earthquake.data.Cluster;

import javax.swing.*;
import java.awt.*;
import java.util.stream.IntStream;

public class AlertSettingsPanel extends SettingsPanel {
    private JCheckBox chkBoxLocal;
    private JTextField textFieldLocalDist;
    private JCheckBox chkBoxRegion;
    private JTextField textFieldRegionMag;
    private JTextField textFieldRegionDist;
    private JCheckBox checkBoxGlobal;
    private JTextField textFieldGlobalMag;
    private JLabel label1;
    private JCheckBox chkBoxFocus;
    private JCheckBox chkBoxJumpToAlert;
    private IntensityScaleSelector shakingThreshold;
    private IntensityScaleSelector strongShakingThreshold;
    private JCheckBox chkBoxPossibleShaking;
    private JTextField textFieldPossibleShakingDistance;
    private JCheckBox chkBoxEarthquakeSounds;
    private JTextField textFieldQuakeMinMag;
    private JTextField textFieldQuakeMaxDist;
    private JLabel label2;
    private IntensityScaleSelector eewThreshold;
    private JComboBox<Integer> comboBoxEEWClusterLevel;

    public AlertSettingsPanel() {
        setLayout(new BorderLayout());

        JTabbedPane tabbedPane = new JTabbedPane();
        tabbedPane.addTab("Warnings", createWarningsTab());
        tabbedPane.addTab("Pings", createPingsTab());

        add(tabbedPane, BorderLayout.CENTER);

        refreshUI();
    }

    private Component createWarningsTab() {
        JPanel panel = new JPanel(new BorderLayout());

        JPanel warningsPanel = createVerticalPanel(false);
        warningsPanel.add(createAlertDialogSettings());
        warningsPanel.add(createAlertLevels());
        panel.add(warningsPanel, BorderLayout.NORTH);

        return panel;
    }

    private Component createAlertDialogSettings() {
        JPanel panel = createGridBagPanel("Alert settings");

        JPanel localPanel = createGridBagPanel("Local area");

        chkBoxLocal = new JCheckBox("", Settings.alertLocal);
        chkBoxLocal.addChangeListener(changeEvent -> textFieldLocalDist.setEnabled(chkBoxLocal.isSelected()));
        localPanel.add(chkBoxLocal, createGbc(0, 0));

        textFieldLocalDist = new JTextField();
        textFieldLocalDist.setEnabled(chkBoxLocal.isSelected());
        localPanel.add(textFieldLocalDist, createGbc(1, 0));

        panel.add(localPanel, createGbc(0));

        JPanel regionPanel = createGridBagPanel("Regional area");

        chkBoxRegion = new JCheckBox("Alert earthquakes larger than (magnitude):", Settings.alertRegion);
        chkBoxRegion.addChangeListener(changeEvent -> {
            textFieldRegionMag.setEnabled(chkBoxRegion.isSelected());
            textFieldRegionDist.setEnabled(chkBoxRegion.isSelected());
        });
        regionPanel.add(chkBoxRegion, createGbc(0, 0));

        textFieldRegionMag = new JTextField(String.valueOf(Settings.alertRegionMag));
        textFieldRegionMag.setEnabled(chkBoxRegion.isSelected());
        regionPanel.add(textFieldRegionMag, createGbc(1, 0));

        regionPanel.add(label1 = new JLabel(), createGbc(0, 1));

        textFieldRegionDist = new JTextField();
        textFieldRegionDist.setEnabled(chkBoxRegion.isSelected());
        regionPanel.add(textFieldRegionDist, createGbc(1, 1));

        panel.add(regionPanel, createGbc(1));

        JPanel globalPanel = createGridBagPanel("Global");

        checkBoxGlobal = new JCheckBox("Alert earthquakes larger than (magnitude):", Settings.alertGlobal);
        checkBoxGlobal.addChangeListener(changeEvent -> textFieldGlobalMag.setEnabled(checkBoxGlobal.isSelected()));
        globalPanel.add(checkBoxGlobal, createGbc(0, 0));

        textFieldGlobalMag = new JTextField(String.valueOf(Settings.alertGlobalMag));
        textFieldGlobalMag.setEnabled(checkBoxGlobal.isSelected());
        globalPanel.add(textFieldGlobalMag, createGbc(1, 0));

        panel.add(globalPanel, createGbc(2));

        chkBoxFocus = new JCheckBox("Focus main window if the conditions above are met.", Settings.focusOnEvent);
        panel.add(chkBoxFocus, createGbc(3));

        chkBoxJumpToAlert = new JCheckBox("Jump directly to the warned event.", Settings.jumpToAlert);
        panel.add(chkBoxJumpToAlert, createGbc(4));

        return panel;
    }

    private Component createAlertLevels() {
        JPanel panel = createGridBagPanel("Alert levels");

        panel.add(alignLeft(shakingThreshold = new IntensityScaleSelector("Shaking alert threshold:", Settings.shakingLevelScale, Settings.shakingLevelIndex)), createGbc(0));
        panel.add(alignLeft(strongShakingThreshold = new IntensityScaleSelector("Strong shaking alert threshold:", Settings.strongShakingLevelScale, Settings.strongShakingLevelIndex)), createGbc(1));

        return panel;
    }

    private Component createPingsTab() {
        JPanel panel = new JPanel(new BorderLayout());

        JPanel pingsPanel = createVerticalPanel(false);
        pingsPanel.add(createPossibleShakingPanel());
        pingsPanel.add(createEarthquakeSoundsPanel());

        JPanel eewThresholdPanel = createGridBagPanel("EEW");

        eewThresholdPanel.add(new JLabel("Trigger eew_warning.wav sound effect if estimated intensity at land reaches:"), createGbc(0));
        eewThresholdPanel.add(alignLeft(eewThreshold = new IntensityScaleSelector(null, Settings.eewScale, Settings.eewLevelIndex)), createGbc(1));

        JPanel maxClusterLevelPanel = new JPanel();
        maxClusterLevelPanel.add(new JLabel("and the associated Cluster has level at least:"));

        comboBoxEEWClusterLevel = new JComboBox<>();
        for (int i : IntStream.rangeClosed(0, Cluster.MAX_LEVEL).toArray()) {
            comboBoxEEWClusterLevel.addItem(i);
        }
        comboBoxEEWClusterLevel.setSelectedIndex(Settings.eewClusterLevel);
        maxClusterLevelPanel.add(comboBoxEEWClusterLevel);
        eewThresholdPanel.add(alignLeft(maxClusterLevelPanel), createGbc(2));

        pingsPanel.add(eewThresholdPanel);
        panel.add(pingsPanel, BorderLayout.NORTH);

        return panel;
    }

    private Component createPossibleShakingPanel() {
        JPanel possibleShakingPanel = createGridBagPanel("Possible shaking detection");

        chkBoxPossibleShaking = new JCheckBox("", Settings.alertPossibleShaking);
        chkBoxPossibleShaking.addChangeListener(changeEvent -> textFieldPossibleShakingDistance.setEnabled(chkBoxPossibleShaking.isSelected()));
        possibleShakingPanel.add(chkBoxPossibleShaking, createGbc(0, 0));

        textFieldPossibleShakingDistance = new JTextField(String.valueOf(Settings.alertPossibleShakingDistance));
        textFieldPossibleShakingDistance.setEnabled(chkBoxPossibleShaking.isSelected());
        possibleShakingPanel.add(textFieldPossibleShakingDistance, createGbc(1, 0));

        return possibleShakingPanel;
    }

    private Component createEarthquakeSoundsPanel() {
        JPanel earthquakePanel = createGridBagPanel("Earthquake alerts");

        chkBoxEarthquakeSounds = new JCheckBox("Play sound alerts if earthquake is bigger than (magnitude):", Settings.enableEarthquakeSounds);
        chkBoxEarthquakeSounds.addChangeListener(changeEvent -> {
            textFieldQuakeMinMag.setEnabled(chkBoxEarthquakeSounds.isSelected());
            textFieldQuakeMaxDist.setEnabled(chkBoxEarthquakeSounds.isSelected());
        });
        earthquakePanel.add(chkBoxEarthquakeSounds, createGbc(0, 0));

        textFieldQuakeMinMag = new JTextField(String.valueOf(Settings.earthquakeSoundsMinMagnitude));
        textFieldQuakeMinMag.setEnabled(chkBoxEarthquakeSounds.isSelected());
        earthquakePanel.add(textFieldQuakeMinMag, createGbc(1, 0));

        earthquakePanel.add(label2 = new JLabel(), createGbc(0, 1));

        textFieldQuakeMaxDist = new JTextField();
        textFieldQuakeMaxDist.setEnabled(chkBoxEarthquakeSounds.isSelected());
        earthquakePanel.add(textFieldQuakeMaxDist, createGbc(1, 1));

        return earthquakePanel;
    }

    @Override
    public void refreshUI() {
        chkBoxLocal.setText("Alert when any earthquake occurs closer than (%s):".formatted(Settings.getSelectedDistanceUnit().getShortName()));
        label1.setText("and are closer from home location than (%s):".formatted(Settings.getSelectedDistanceUnit().getShortName()));
        label2.setText("or is closer from home location than (%s):".formatted(Settings.getSelectedDistanceUnit().getShortName()));
        chkBoxPossibleShaking.setText("Play sound if possible shaking is detected closer than (%s):".formatted(Settings.getSelectedDistanceUnit().getShortName()));

        textFieldLocalDist.setText(String.format("%.1f", Settings.alertLocalDist * Settings.getSelectedDistanceUnit().getKmRatio()));
        textFieldRegionDist.setText(String.format("%.1f", Settings.alertRegionDist * Settings.getSelectedDistanceUnit().getKmRatio()));
        textFieldPossibleShakingDistance.setText(String.format("%.1f", Settings.alertPossibleShakingDistance * Settings.getSelectedDistanceUnit().getKmRatio()));
        textFieldQuakeMaxDist.setText(String.format("%.1f", Settings.earthquakeSoundsMaxDist * Settings.getSelectedDistanceUnit().getKmRatio()));

        revalidate();
        repaint();
    }

    @Override
    public void save() throws NumberFormatException {
        Settings.alertLocal = chkBoxLocal.isSelected();
        Settings.alertLocalDist = parseDouble(textFieldLocalDist.getText(), "Local alert distance", 0, 30000) / Settings.getSelectedDistanceUnit().getKmRatio();
        Settings.alertRegion = chkBoxRegion.isSelected();
        Settings.alertRegionMag = parseDouble(textFieldRegionMag.getText(), "Regional alert Magnitude", 0, 10);
        Settings.alertRegionDist = parseDouble(textFieldRegionDist.getText(), "Regional alert distance", 0, 30000) / Settings.getSelectedDistanceUnit().getKmRatio();

        Settings.alertGlobal = checkBoxGlobal.isSelected();
        Settings.alertGlobalMag = parseDouble(textFieldGlobalMag.getText(), "Global alert magnitude", 0, 10);
        Settings.focusOnEvent = chkBoxFocus.isSelected();
        Settings.jumpToAlert = chkBoxJumpToAlert.isSelected();

        Settings.shakingLevelScale = shakingThreshold.getShakingScaleComboBox().getSelectedIndex();
        Settings.shakingLevelIndex = shakingThreshold.getLevelComboBox().getSelectedIndex();

        Settings.strongShakingLevelScale = strongShakingThreshold.getShakingScaleComboBox().getSelectedIndex();
        Settings.strongShakingLevelIndex = strongShakingThreshold.getLevelComboBox().getSelectedIndex();

        Settings.alertPossibleShaking = chkBoxPossibleShaking.isSelected();
        Settings.alertPossibleShakingDistance = parseDouble(textFieldPossibleShakingDistance.getText(), "Possible shaking alert radius", 0, 30000) / Settings.getSelectedDistanceUnit().getKmRatio();
        Settings.enableEarthquakeSounds = chkBoxEarthquakeSounds.isSelected();
        Settings.earthquakeSoundsMinMagnitude = parseDouble(textFieldQuakeMinMag.getText(), "Earthquake minimum magnitude to play sound", 0, 10);
        Settings.earthquakeSoundsMaxDist = parseDouble(textFieldQuakeMaxDist.getText(), "Earthquake maximum distance to play sound", 0, 30000) / Settings.getSelectedDistanceUnit().getKmRatio();

        Settings.eewScale = eewThreshold.getShakingScaleComboBox().getSelectedIndex();
        Settings.eewLevelIndex = eewThreshold.getLevelComboBox().getSelectedIndex();
        Settings.eewClusterLevel = (Integer) comboBoxEEWClusterLevel.getSelectedItem();
    }

    @Override
    public String getTitle() {
        return "Alerts";
    }
}