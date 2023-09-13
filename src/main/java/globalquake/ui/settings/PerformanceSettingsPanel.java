package globalquake.ui.settings;

import globalquake.training.EarthquakeAnalysisTraining;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;

public class PerformanceSettingsPanel extends SettingsPanel {
    private JSlider sliderResolution;
    private JCheckBox chkBoxParalell;
    private JSlider sliderStoreTime;

    public PerformanceSettingsPanel() {
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));

        add(createSettingAccuracy());
        add(createSettingStoreTime());
        add(createSettingParalell());

        for(int i = 0; i < 3; i++){
            add(new JPanel()); // fillers
        }
    }

    private Component createSettingStoreTime() {
        sliderStoreTime = HypocenterAnalysisSettingsPanel.createSettingsSlider(2, 20, 2, 1);

        JLabel label = new JLabel();
        ChangeListener changeListener = changeEvent -> label.setText("Waveform data storage time (minutes): %d".formatted(
                sliderStoreTime.getValue()));

        sliderStoreTime.addChangeListener(changeListener);

        sliderStoreTime.setValue(Settings.logsStoreTimeMinutes);
        changeListener.stateChanged(null);

        return HypocenterAnalysisSettingsPanel.createCoolLayout(sliderStoreTime, label, "5",
                """
                        In GlobalQuake, waveform data poses the highest demand on your system's RAM.
                        If you're encountering memory constraints, you have two options:
                        either reduce the number of selected stations or lower this specific value.
                        """);
    }

    @SuppressWarnings("ExtractMethodRecommender")
    private JPanel createSettingParalell() {
        JPanel panel = new JPanel();
        panel.setBorder(BorderFactory.createRaisedBevelBorder());
        panel.setLayout(new BorderLayout());
        chkBoxParalell = new JCheckBox("Use all CPU cores");
        chkBoxParalell.setSelected(Settings.parallelHypocenterLocations);

        JTextArea textAreaExplanation = new JTextArea(
                """
                        Using all CPU cores will make the Hypocenter Finding much faster,\s
                        but it will be using 100% of your CPU, which can increase lags.
                        Make sure you select the optimal resolution above for your system.""");
        textAreaExplanation.setBorder(new EmptyBorder(5,5,5,5));
        textAreaExplanation.setEditable(false);
        textAreaExplanation.setBackground(panel.getBackground());

        chkBoxParalell.addChangeListener(changeEvent -> Settings.parallelHypocenterLocations = chkBoxParalell.isSelected());

        panel.add(chkBoxParalell, BorderLayout.CENTER);
        panel.add(textAreaExplanation, BorderLayout.SOUTH);
        return panel;
    }

    @Override
    public void save() {
        Settings.hypocenterDetectionResolution = (double) sliderResolution.getValue();
        Settings.parallelHypocenterLocations = chkBoxParalell.isSelected();
        Settings.logsStoreTimeMinutes = sliderStoreTime.getValue();
    }

    private Component createSettingAccuracy() {
        sliderResolution = HypocenterAnalysisSettingsPanel.createSettingsSlider(0, (int) Settings.hypocenterDetectionResolutionMax, 10, 5);

        JLabel label = new JLabel();
        ChangeListener changeListener = changeEvent -> label.setText("Hypocenter Finding Resolution: %.2f ~ %s".formatted(
                sliderResolution.getValue() / 100.0,
                getNameForResolution(sliderResolution.getValue())));

        sliderResolution.addChangeListener(changeListener);

        sliderResolution.setValue(Settings.hypocenterDetectionResolution.intValue());
        changeListener.stateChanged(null);

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
                    EarthquakeAnalysisTraining.calibrateResolution(null, sliderResolution);
                    btnRecalibrate.setEnabled(true);
                    sliderResolution.setEnabled(true);
                }).start();
             }
        });

        panel2.add(btnRecalibrate);
        panel.add(panel2, BorderLayout.SOUTH);

        return panel;
    }

    public static final String[] RESOLUTION_NAMES = {"Very Low", "Low", "Decreased", "Default", "Increased", "High", "Very High", "Extremely High", "Insane"};

    private String getNameForResolution(int value) {
        return RESOLUTION_NAMES[(int) Math.max(0, Math.min(RESOLUTION_NAMES.length - 1, ((value / Settings.hypocenterDetectionResolutionMax) * (RESOLUTION_NAMES.length))))];
    }

    @Override
    public String getTitle() {
        return "Performance";
    }
}
