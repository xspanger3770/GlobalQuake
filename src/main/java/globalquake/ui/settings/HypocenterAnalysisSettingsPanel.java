package globalquake.ui.settings;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.event.ChangeListener;
import java.awt.*;

public class HypocenterAnalysisSettingsPanel extends SettingsPanel {

    private JSlider sliderPWaveInaccuracy;
    private JSlider sliderCorrectness;
    private JSlider sliderMinStations;

    public HypocenterAnalysisSettingsPanel() {
        setLayout(new BorderLayout());

        JPanel contentPanel = new JPanel();
        contentPanel.setLayout(new BoxLayout(contentPanel, BoxLayout.Y_AXIS));

        contentPanel.add(createMinStationsSetting());
        contentPanel.add(createSettingPWave());
        contentPanel.add(createSettingCorrectness());

        JScrollPane scrollPane = new JScrollPane(contentPanel);
        scrollPane.setPreferredSize(new Dimension(300, 300));

        javax.swing.SwingUtilities.invokeLater(() -> scrollPane.getVerticalScrollBar().setValue(0));

        add(scrollPane);
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

    public static Component createCoolLayout(JSlider slider, JLabel label, String defaultValue, String explanation){
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBorder(BorderFactory.createRaisedBevelBorder());

        JPanel topPanel = new JPanel(new BorderLayout());
        topPanel.setBorder(new EmptyBorder(5,5,5,5));

        topPanel.add(label, BorderLayout.NORTH);
        topPanel.add(slider, BorderLayout.CENTER);

        if(defaultValue != null) {
            JLabel labelDefault = new JLabel("Default value: " + defaultValue);
            labelDefault.setBorder(new EmptyBorder(8, 2, 0, 0));
            topPanel.add(labelDefault, BorderLayout.SOUTH);
        }

        JTextArea textAreaExplanation = new JTextArea(explanation);
        textAreaExplanation.setBorder(new EmptyBorder(5,5,5,5));
        textAreaExplanation.setEditable(false);
        textAreaExplanation.setBackground(panel.getBackground());

        panel.add(topPanel, BorderLayout.NORTH);
        panel.add(textAreaExplanation, BorderLayout.CENTER);

        return panel;
    }

    public static JSlider createSettingsSlider(int min, int max, int major, int minor){
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
        sliderPWaveInaccuracy = createSettingsSlider(400, 2500, 200, 100);

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
    }

    @Override
    public String getTitle() {
        return "Advanced";
    }
}
