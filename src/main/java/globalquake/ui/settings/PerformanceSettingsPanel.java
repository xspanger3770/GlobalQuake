package globalquake.ui.settings;

import javax.swing.*;
import java.awt.*;

public class PerformanceSettingsPanel extends SettingsPanel {
    private JSlider sliderResolution;

    public PerformanceSettingsPanel() {
        setLayout(new BorderLayout());
        setPreferredSize(new Dimension(400, 300));

        JPanel contentPanel = new JPanel();
        contentPanel.setLayout(new BoxLayout(contentPanel, BoxLayout.Y_AXIS));

        contentPanel.add(createSettingAccuracy());

        add(new JScrollPane(contentPanel), BorderLayout.CENTER);
    }

    @Override
    public void save() {
        Settings.hypocenterDetectionResolution = (double) sliderResolution.getValue();
    }

    private Component createSettingAccuracy() {
        sliderResolution = HypocenterAnalysisSettingsPanel.createSettingsSlider(0, 100, 10, 2);
        sliderResolution.setPaintLabels(false);

        JLabel label = new JLabel();
        sliderResolution.addChangeListener(changeEvent -> label.setText("Hypocenter Finding Resolution: %.2f ~ %s".formatted(
                sliderResolution.getValue() / 100.0,
                getNameForResolution(sliderResolution.getValue()))));

        sliderResolution.setValue(Settings.hypocenterDetectionResolution.intValue());

        return HypocenterAnalysisSettingsPanel.createCoolLayout(sliderResolution, label, Settings.hypocenterDetectionResolutionDefault+"",
                """
                        By increasing the Hypocenter Finding Resolution, you can\s
                        enhance the accuracy at which GlobalQuake detects hypocenters
                        at the cost of increased demand on your CPU. If you experience
                        significant lags while there is an earthquake happening on the map,
                        you should decrease this value.
                        """);
    }

    public static final String[] RESOLUTION_NAMES = {"Very Low", "Low", "Default", "Increased", "High", "Very High", "Insane"};

    private String getNameForResolution(int value) {
        return RESOLUTION_NAMES[(int) Math.max(0, Math.min(RESOLUTION_NAMES.length - 1, ((value / 100.0) * (RESOLUTION_NAMES.length))))];
    }

    @Override
    public String getTitle() {
        return "Performance";
    }
}
