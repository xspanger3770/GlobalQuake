package globalquake.ui.settings;

import javax.swing.*;
import javax.swing.border.EmptyBorder;

public class GraphicsSettingsPanel extends SettingsPanel{

    private final JCheckBox chkBoxScheme;
    private final JCheckBox chkBoxHomeLoc;
    private final JCheckBox chkBoxAntialiasing;
    private JSlider sliderFpsIdle;

    public GraphicsSettingsPanel() {
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setBorder(new EmptyBorder(5,5,5,5));

        createFpsSlider();

        chkBoxScheme = new JCheckBox("Use old color scheme (exaggerated)");
        chkBoxScheme.setSelected(Settings.useOldColorScheme);
        add(chkBoxScheme);

        chkBoxHomeLoc = new JCheckBox("Display home location");
        chkBoxHomeLoc.setSelected(Settings.displayHomeLocation);
        add(chkBoxHomeLoc);

        chkBoxAntialiasing = new JCheckBox("Enable antialiasing for stations");
        chkBoxAntialiasing.setSelected(Settings.antialiasing);
        add(chkBoxAntialiasing);
    }

    private void createFpsSlider() {
        sliderFpsIdle = new JSlider(JSlider.HORIZONTAL, 10, 90, Settings.fpsIdle);
        sliderFpsIdle.setPaintLabels(true);
        sliderFpsIdle.setPaintTicks(true);
        sliderFpsIdle.setMajorTickSpacing(10);
        sliderFpsIdle.setMinorTickSpacing(5);
        sliderFpsIdle.setBorder(new EmptyBorder(5,5,10,5));

        JLabel label = new JLabel("FPS at idle: "+sliderFpsIdle.getValue());

        sliderFpsIdle.addChangeListener(changeEvent -> label.setText("FPS at idle: "+sliderFpsIdle.getValue()));

        add(label);
        add(sliderFpsIdle);
    }

    @Override
    public void save() {
        Settings.useOldColorScheme = chkBoxScheme.isSelected();
        Settings.displayHomeLocation = chkBoxHomeLoc.isSelected();
        Settings.antialiasing = chkBoxAntialiasing.isSelected();
        Settings.fpsIdle = sliderFpsIdle.getValue();
    }

    @Override
    public String getTitle() {
        return "Graphics";
    }
}
