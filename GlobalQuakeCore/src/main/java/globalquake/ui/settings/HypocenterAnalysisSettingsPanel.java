package globalquake.ui.settings;

import globalquake.core.Settings;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.event.ChangeListener;
import java.awt.*;

public class HypocenterAnalysisSettingsPanel extends SettingsPanel {

    private JSlider sliderPWaveInaccuracy;
    private JSlider sliderCorrectness;
    private JSlider sliderMinStations;
    private JSlider sliderMaxStations;

    public HypocenterAnalysisSettingsPanel() {
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));

        add(createMinStationsSetting());
        add(createMaxStationsSetting());
        add(createSettingPWave());
        add(createSettingCorrectness());
    }

    private Component createMaxStationsSetting() {
        sliderMaxStations = createSettingsSlider(20, 300, 20, 5);

        JLabel label = new JLabel();

        ChangeListener upd = changeEvent -> label.setText("Maximum number of associated stations: %d".formatted(sliderMaxStations.getValue()));

        sliderMaxStations.addChangeListener(upd);
        sliderMaxStations.setValue(Settings.maxEvents);

        upd.stateChanged(null);

        return createCoolLayout(sliderMaxStations, label, "%s".formatted(Settings.maxEventsDefault),
                """
                        Here you can set the maximum number of stations that will\s
                        be used for the hypocenter finding algorithm.\s
                        Increasing this value can enhance the accuracy of earthquake detection,\s
                        however, it's important to note that doing so will also significantly\s
                        escalate the computational demands, potentially leading to longer processing times.
                        """);
    }

    private Component createMinStationsSetting() {
        sliderMinStations = createSettingsSlider(4, 16, 1, 1);

        JLabel label = new JLabel();

        ChangeListener upd = changeEvent -> label.setText("Minimum number of stations: %d".formatted(sliderMinStations.getValue()));

        sliderMinStations.addChangeListener(upd);
        sliderMinStations.setValue(Settings.minimumStationsForEEW);

        upd.stateChanged(null);

        return createCoolLayout(sliderMinStations, label, "%s".formatted(Settings.minimumStationsForEEWDefault),
                """
                        Here you can set the minimum number of stations that need to pick\s
                        up the earthquake for the EEW to be issued.\s
                        Increasing the number can greatly reduce the number of false alarms,\s
                        but may also cause EEW's to not appear in areas with less stations.
                        """);
    }

    public static JPanel createCoolLayout(JSlider slider, JLabel label, String defaultValue, String explanation) {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBorder(BorderFactory.createRaisedBevelBorder());

        JPanel topPanel = new JPanel(new BorderLayout());
        topPanel.setBorder(new EmptyBorder(5, 5, 5, 5));

        topPanel.add(label, BorderLayout.NORTH);
        topPanel.add(slider, BorderLayout.CENTER);

        if (defaultValue != null) {
            JLabel labelDefault = new JLabel("Default value: " + defaultValue);
            labelDefault.setBorder(new EmptyBorder(8, 2, 0, 0));
            topPanel.add(labelDefault, BorderLayout.SOUTH);
        }

        if (explanation != null) {
            JTextArea textAreaExplanation = new JTextArea(explanation);
            textAreaExplanation.setBorder(new EmptyBorder(5, 5, 5, 5));
            textAreaExplanation.setEditable(false);
            textAreaExplanation.setBackground(panel.getBackground());
            panel.add(textAreaExplanation, BorderLayout.CENTER);
        }

        panel.add(topPanel, BorderLayout.NORTH);

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

    private Component createSettingCorrectness() {
        sliderCorrectness = createSettingsSlider(20, 90, 10, 2);

        JLabel label = new JLabel();

        ChangeListener upd = changeEvent -> label.setText("Hypocenter Correctness Threshold: %d %%".formatted(sliderCorrectness.getValue()));

        sliderCorrectness.addChangeListener(upd);
        sliderCorrectness.setValue(Settings.hypocenterCorrectThreshold.intValue());

        upd.stateChanged(null);

        return createCoolLayout(sliderCorrectness, label, "%s %%".formatted(Settings.hypocenterCorrectThresholdDefault),
                """
                        This value determines the threshold when a hypocenter is considered
                        correct or not.
                        The correctness is calculated as the % of stations that have arrival
                        within the Inaccuracy threshold and total number of stations used by
                        the hypocenter locating algoritgm.
                        If hypocenter is marked as incorrect, the earthquake will not
                        be displayed on the map.
                        Higher values will lead to more false negatives,
                        Lower values will lead to more false positives.
                        """);
    }

    private Component createSettingPWave() {
        sliderPWaveInaccuracy = createSettingsSlider(400, 5200, 400, 200);

        JLabel label = new JLabel();
        ChangeListener changeListener = changeEvent -> label.setText("P Wave Arrival Inaccuracy Threshold: %d ms".formatted(sliderPWaveInaccuracy.getValue()));
        sliderPWaveInaccuracy.addChangeListener(changeListener);

        sliderPWaveInaccuracy.setValue(Settings.pWaveInaccuracyThreshold.intValue());
        changeListener.stateChanged(null);

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
