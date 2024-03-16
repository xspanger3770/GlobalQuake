package globalquake.ui.settings;

import globalquake.core.Settings;

import javax.swing.*;
import java.awt.*;

public class HypocenterAnalysisSettingsPanel extends SettingsPanel {

    private JSlider sliderPWaveInaccuracy;
    private JSlider sliderCorrectness;
    private JSlider sliderMinStations;
    private JSlider sliderMaxStations;

    public HypocenterAnalysisSettingsPanel() {
        setLayout(new BorderLayout());

        JPanel panel = createVerticalPanel(false);
        panel.add(createMinStationsSetting());
        panel.add(createMaxStationsSetting());
        panel.add(createSettingPWave());
        panel.add(createSettingCorrectness());

        add(panel, BorderLayout.NORTH);
    }

    private Component createMinStationsSetting() {
        sliderMinStations = createSettingsSlider(4, 16, 1, 1);

        JLabel label = new JLabel("Minimum number of stations: %d".formatted(sliderMinStations.getValue()));

        sliderMinStations.addChangeListener(e -> label.setText("Minimum number of stations: %d".formatted(sliderMinStations.getValue())));
        sliderMinStations.setValue(Settings.minimumStationsForEEW);

        return createCoolLayout(sliderMinStations, label, "%s".formatted(Settings.minimumStationsForEEWDefault),
                """
                        Here you can set the minimum number of stations that need to pick\s
                        up the earthquake for the EEW to be issued.\s
                        Increasing the number can greatly reduce the number of false alarms,\s
                        but may also cause EEW's to not appear in areas with less stations.
                        """);
    }

    private Component createMaxStationsSetting() {
        sliderMaxStations = createSettingsSlider(20, 300, 20, 5);

        JLabel label = new JLabel("Maximum number of associated stations: %d".formatted(sliderMaxStations.getValue()));

        sliderMaxStations.addChangeListener(e -> label.setText("Maximum number of associated stations: %d".formatted(sliderMaxStations.getValue())));
        sliderMaxStations.setValue(Settings.maxEvents);

        return createCoolLayout(sliderMaxStations, label, "%s".formatted(Settings.maxEventsDefault),
                """
                        Here you can set the maximum number of stations that will\s
                        be used for the hypocenter finding algorithm.\s
                        Increasing this value can enhance the accuracy of earthquake detection,\s
                        however, it's important to note that doing so will also significantly\s
                        escalate the computational demands, potentially leading to longer processing times.
                        """);
    }

    private Component createSettingPWave() {
        sliderPWaveInaccuracy = createSettingsSlider(400, 5200, 400, 200);

        JLabel label = new JLabel("P Wave Arrival Inaccuracy Threshold: %d ms".formatted(sliderPWaveInaccuracy.getValue()));
        sliderPWaveInaccuracy.addChangeListener(e -> label.setText("P Wave Arrival Inaccuracy Threshold: %d ms".formatted(sliderPWaveInaccuracy.getValue())));

        sliderPWaveInaccuracy.setValue(Settings.pWaveInaccuracyThreshold.intValue());

        return createCoolLayout(sliderPWaveInaccuracy, label, "%s ms".formatted(Settings.pWaveInaccuracyThresholdDefault),
                """
                        This value determines the threshold value when the hypocenter finding\s
                        algorithm considers the arrival from current point to a station correct \s
                        or incorrect\s
                        Higher values are less restrictive and will lead to more false positives.
                        Lower values will force the algorithm to find more accurate hypocenter \s
                        and will lead to more false negatives.
                        """);
    }

    private Component createSettingCorrectness() {
        sliderCorrectness = createSettingsSlider(20, 90, 10, 2);

        JLabel label = new JLabel("Hypocenter Correctness Threshold: %d %%".formatted(sliderCorrectness.getValue()));

        sliderCorrectness.addChangeListener(e -> label.setText("Hypocenter Correctness Threshold: %d %%".formatted(sliderCorrectness.getValue())));
        sliderCorrectness.setValue(Settings.hypocenterCorrectThreshold.intValue());

        return createCoolLayout(sliderCorrectness, label, "%s %%".formatted(Settings.hypocenterCorrectThresholdDefault),
                """
                        This value determines the threshold when a hypocenter is considered
                        correct or not.
                        The correctness is calculated as the % of stations that have arrival
                        within the Inaccuracy threshold and total number of stations used by
                        the hypocenter locating algorithm.
                        If hypocenter is marked as incorrect, the earthquake will not
                        be displayed on the map.
                        Higher values will lead to more false negatives,
                        Lower values will lead to more false positives.
                        """);
    }

    public static JPanel createCoolLayout(JSlider slider, JLabel label, String defaultValue, String explanation) {
        JPanel panel = createGridBagPanel();
        int row = 0;

        panel.add(label, createGbc(row++));
        panel.add(slider, createGbc(row++));

        if (defaultValue != null) {
            panel.add(new JLabel("Default value: %s".formatted(defaultValue)), createGbc(row++));
        }

        if (explanation != null) {
            panel.add(createJTextArea(explanation, panel), createGbc(row));
        }

        return panel;
    }

    public static JSlider createSettingsSlider(int min, int max, int major, int minor) {
        JSlider slider = new JSlider();
        slider.setMinimum(min);
        slider.setMaximum(max);
        slider.setMajorTickSpacing(major);
        slider.setMinorTickSpacing(minor);

        slider.setPaintLabels(true);
        slider.setPaintTicks(true);
        return slider;
    }

    @Override
    public void save() {
        Settings.pWaveInaccuracyThreshold = (double) sliderPWaveInaccuracy.getValue();
        Settings.hypocenterCorrectThreshold = (double) sliderCorrectness.getValue();
        Settings.minimumStationsForEEW = sliderMinStations.getValue();
        Settings.maxEvents = sliderMaxStations.getValue();
    }

    @Override
    public String getTitle() {
        return "Advanced";
    }
}
