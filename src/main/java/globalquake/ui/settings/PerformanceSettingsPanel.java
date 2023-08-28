package globalquake.ui.settings;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.event.ChangeListener;
import java.awt.*;

public class PerformanceSettingsPanel extends SettingsPanel {
    private static final double RESOLUTION_MAX = 100.0;
    private JSlider sliderResolution;
    private JCheckBox chkBoxParalell;

    public PerformanceSettingsPanel() {
        setLayout(new BorderLayout());
        setPreferredSize(new Dimension(400, 300));

        JPanel contentPanel = new JPanel();
        contentPanel.setLayout(new BoxLayout(contentPanel, BoxLayout.Y_AXIS));

        contentPanel.add(createSettingAccuracy());
        contentPanel.add(createSettingParalell());

        add(new JScrollPane(contentPanel), BorderLayout.CENTER);
    }

    private JPanel createSettingParalell() {
        JPanel panel = new JPanel();
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

        panel.add(chkBoxParalell, BorderLayout.CENTER);
        panel.add(textAreaExplanation, BorderLayout.SOUTH);
        return panel;
    }

    @Override
    public void save() {
        Settings.hypocenterDetectionResolution = (double) sliderResolution.getValue();
        Settings.parallelHypocenterLocations = chkBoxParalell.isSelected();
    }

    private Component createSettingAccuracy() {
        sliderResolution = HypocenterAnalysisSettingsPanel.createSettingsSlider(0, (int) RESOLUTION_MAX, 20, 5);
        sliderResolution.setPaintLabels(false);

        JLabel label = new JLabel();
        ChangeListener changeListener = changeEvent -> label.setText("Hypocenter Finding Resolution: %.2f ~ %s".formatted(
                sliderResolution.getValue() / 100.0,
                getNameForResolution(sliderResolution.getValue())));

        sliderResolution.addChangeListener(changeListener);

        sliderResolution.setValue(Settings.hypocenterDetectionResolution.intValue());
        changeListener.stateChanged(null);

        return HypocenterAnalysisSettingsPanel.createCoolLayout(sliderResolution, label, "%.2f".formatted(Settings.hypocenterDetectionResolutionDefault / 100.0),
                """
                        By increasing the Hypocenter Finding Resolution, you can\s
                        enhance the accuracy at which GlobalQuake locates hypocenters
                        at the cost of increased demand on your CPU. If you experience
                        significant lags while there is an earthquake happening on the map,
                        you should decrease this value.
                        """);
    }

    public static final String[] RESOLUTION_NAMES = {"Very Low", "Low", "Decreased", "Default", "Increased", "High", "Very High", "Insane"};

    private String getNameForResolution(int value) {
        return RESOLUTION_NAMES[(int) Math.max(0, Math.min(RESOLUTION_NAMES.length - 1, ((value / RESOLUTION_MAX) * (RESOLUTION_NAMES.length))))];
    }

    @Override
    public String getTitle() {
        return "Performance";
    }
}
