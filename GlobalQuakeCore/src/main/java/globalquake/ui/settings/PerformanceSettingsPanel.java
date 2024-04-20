package globalquake.ui.settings;

import globalquake.core.Settings;
import globalquake.core.training.EarthquakeAnalysisTraining;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;

public class PerformanceSettingsPanel extends SettingsPanel {
    private static final String[] RESOLUTION_NAMES = {"Very Low", "Low", "Default", "Increased", "High", "Very High", "Extremely High", "Insane"};

    private JSlider sliderResolution;
    private JCheckBox chkBoxParallel;
    private JCheckBox chkBoxRecalibrateOnLaunch;

    public PerformanceSettingsPanel() {
        setLayout(new BorderLayout());

        JPanel panel = createVerticalPanel(false);
        panel.add(createSettingAccuracy());
        panel.add(createSettingParallel());

        add(panel, BorderLayout.NORTH);
    }

    private Component createSettingAccuracy() {
        sliderResolution = HypocenterAnalysisSettingsPanel.createSettingsSlider(0, 160, 10, 5);

        JLabel label = new JLabel("Hypocenter Finding Resolution (CPU): %.2f ~ %s".formatted(
                sliderResolution.getValue() / 100.0,
                getNameForResolution(sliderResolution.getValue())));

        sliderResolution.setValue(Settings.hypocenterDetectionResolution.intValue());
        sliderResolution.addChangeListener(e -> label.setText("Hypocenter Finding Resolution (CPU): %.2f ~ %s".formatted(
                sliderResolution.getValue() / 100.0, getNameForResolution(sliderResolution.getValue()))));

        JPanel panel = HypocenterAnalysisSettingsPanel.createCoolLayout(sliderResolution, label, "%.2f".formatted(Settings.hypocenterDetectionResolutionDefault / 100.0),
                """
                        By increasing the Hypocenter Finding Resolution, you can\s
                        enhance the accuracy at which GlobalQuake locates hypocenters
                        at the cost of increased demand on your CPU. If you experience
                        significant lags while there is an earthquake happening on the map,
                        you should decrease this value.
                        """);

        JPanel panel2 = new JPanel();

        JButton btnRecalibrate = new JButton("Recalibrate");
        btnRecalibrate.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                btnRecalibrate.setEnabled(false);
                sliderResolution.setEnabled(false);
                new Thread(() -> {
                    EarthquakeAnalysisTraining.calibrateResolution(null, sliderResolution, true);
                    btnRecalibrate.setEnabled(true);
                    sliderResolution.setEnabled(true);
                }).start();
            }
        });

        panel2.add(btnRecalibrate);

        JButton testSpeed = new JButton("Test Hypocenter Search");
        testSpeed.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                testSpeed.setEnabled(false);
                new Thread(() -> {
                    testSpeed.setText("Test took %d ms".formatted(EarthquakeAnalysisTraining.measureTest(System.currentTimeMillis(), 60, true)));
                    testSpeed.setEnabled(true);
                }).start();
            }
        });
        panel2.add(testSpeed);

        panel2.add(chkBoxRecalibrateOnLaunch = new JCheckBox("Recalibrate on startup.", Settings.recalibrateOnLaunch));

        panel.add(panel2, createGbc(4));

        return panel;
    }

    private JPanel createSettingParallel() {
        JPanel panel = createGridBagPanel();

        panel.add(chkBoxParallel = new JCheckBox("Use all CPU cores.", Settings.parallelHypocenterLocations), createGbc(0));

        panel.add(createJTextArea("""
                Using all CPU cores will make the Hypocenter Finding much faster,\s
                but it will be using 100% of your CPU, which can increase lags.
                Make sure you select the optimal resolution above for your system.""", panel), createGbc(1));

        return panel;
    }

    @Override
    public void save() {
        Settings.hypocenterDetectionResolution = (double) sliderResolution.getValue();
        Settings.parallelHypocenterLocations = chkBoxParallel.isSelected();
        Settings.recalibrateOnLaunch = chkBoxRecalibrateOnLaunch.isSelected();
    }

    private String getNameForResolution(int value) {
        return RESOLUTION_NAMES[(int) Math.max(0, Math.min(RESOLUTION_NAMES.length - 1, ((value / 160.0) * (RESOLUTION_NAMES.length))))];
    }

    @Override
    public String getTitle() {
        return "Performance";
    }
}
