package globalquake.ui.settings;

import javax.swing.*;
import javax.swing.event.ChangeListener;
import java.awt.*;

public class PerformanceSettingsPanel extends SettingsPanel {
    private static final double RESOLUTION_MAX = 100.0;
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
        sliderResolution = HypocenterAnalysisSettingsPanel.createSettingsSlider(0, (int) RESOLUTION_MAX, 20, 5);
        sliderResolution.setPaintLabels(false);

        JLabel label = new JLabel();
        ChangeListener changeListener = changeEvent -> label.setText("Hypocenter Finding Resolution: %.2f ~ %s".formatted(
                sliderResolution.getValue() / 100.0,
                getNameForResolution(sliderResolution.getValue())));

        sliderResolution.addChangeListener(changeListener);

        sliderResolution.setValue(Settings.hypocenterDetectionResolution.intValue());
        changeListener.stateChanged(null);

        return HypocenterAnalysisSettingsPanel.createCoolLayout(sliderResolution, label, Settings.hypocenterDetectionResolutionDefault+"",
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
