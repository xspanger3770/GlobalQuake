package globalquake.ui.settings;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;

public class HypocenterAnalysisSettingsPanel extends SettingsPanel {

    private JSlider sliderPWaveInaccuracy;
    private JSlider sliderCorrectness;

    public HypocenterAnalysisSettingsPanel() {
        setLayout(new BorderLayout());
        setPreferredSize(new Dimension(400, 300));

        JPanel contentPanel = new JPanel();
        contentPanel.setLayout(new BoxLayout(contentPanel, BoxLayout.Y_AXIS));

        contentPanel.add(createSettingPWave());
        contentPanel.add(createSettingCorrectness());

        add(new JScrollPane(contentPanel), BorderLayout.CENTER);
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
        sliderCorrectness.addChangeListener(changeEvent -> label.setText("Hypocenter correctness treshold: %d %%".formatted(sliderCorrectness.getValue())));

        sliderCorrectness.setValue(Settings.hypocenterCorrectTreshold.intValue());

        return createCoolLayout(sliderCorrectness, label, "%s %%".formatted(Settings.hypocenterCorrectTresholdDefault),
                """
                        This value determines the treshold when a hypocenter is considered
                        correct or not.
                        The correctness is calculated as the % of stations that have arrival
                        within the Inaccuracy treshold and total number of stations used by
                        the hypocenter locating algoritgm.
                        If hypocenter is marked as incorrect, the earthquake will not
                        be displayed on the map.
                        Higher values will lead to more false negatives,
                        Lower values will lead to more false positives.
                        """);
    }

    private Component createSettingPWave() {
        sliderPWaveInaccuracy = createSettingsSlider(500, 3000, 1000, 100);

        JLabel label = new JLabel();
        sliderPWaveInaccuracy.addChangeListener(changeEvent -> label.setText("P Wave Arrival Inaccuracy Treshold: %d ms".formatted(sliderPWaveInaccuracy.getValue())));

        sliderPWaveInaccuracy.setValue(Settings.pWaveInaccuracyTreshold.intValue());

        return createCoolLayout(sliderPWaveInaccuracy, label, "%s ms".formatted(Settings.pWaveInaccuracyTresholdDefault),
                """
                This value determines the treshold value when the hypocenter finding\s
                algorithm considers the arrival from current point to a station correct \s
                or incorrect\s
                Higher values are less restrictive and will lead to more false positives.
                Lower values will force the algorithm to find more accurate hypocenter \s
                and will lead to more false negatives.
                """);
    }

    @Override
    public void save() {
        Settings.pWaveInaccuracyTreshold = (double) sliderPWaveInaccuracy.getValue();
        Settings.hypocenterCorrectTreshold = (double) sliderCorrectness.getValue();
    }

    @Override
    public String getTitle() {
        return "Advanced";
    }
}
