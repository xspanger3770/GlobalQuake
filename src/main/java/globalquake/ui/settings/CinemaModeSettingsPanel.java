package globalquake.ui.settings;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.text.ParseException;

public class CinemaModeSettingsPanel extends SettingsPanel {

    private final JTextField textFieldTime;
    private final JSlider sliderZoomMul;

    private final JCheckBox chkBoxEnableOnStartup;
    private final JCheckBox chkBoxReenable;

    public CinemaModeSettingsPanel() {
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setBorder(new EmptyBorder(5,5,5,5));

        textFieldTime = new JTextField(String.valueOf(Settings.cinemaModeSwitchTime), 12);

        JPanel timePanel = new JPanel();
        timePanel.setLayout(new BoxLayout(timePanel, BoxLayout.X_AXIS));
        timePanel.add(new JLabel("Switch to another point of interest after (seconds): "));
        timePanel.add(textFieldTime);
        add(timePanel);

        JPanel zoomPanel = new JPanel();
        zoomPanel.setBorder(new EmptyBorder(5,5,5,5));

        zoomPanel.setLayout(new BoxLayout(zoomPanel, BoxLayout.X_AXIS));
        zoomPanel.add(new JLabel("Zoom multiplier (move right to zoom in):"));

        sliderZoomMul = new JSlider(JSlider.HORIZONTAL, 20,180, Settings.cinemaModeZoomMultiplier);
        sliderZoomMul.setMinorTickSpacing(5);
        sliderZoomMul.setMajorTickSpacing(20);
        sliderZoomMul.setPaintTicks(true);

        zoomPanel.add(sliderZoomMul);
        add(zoomPanel);

        JPanel chkboxPanel = new JPanel();

        chkboxPanel.add(chkBoxEnableOnStartup = new JCheckBox("Enable Cinema Mode on startup", Settings.cinemaModeOnStartup));
        chkboxPanel.add(chkBoxReenable = new JCheckBox("Re-enable Cinema Mode automatically", Settings.cinemaModeReenable));
        add(chkboxPanel);

        for(int i = 0; i < 39; i++){
            add(new JPanel()); // fillers
        }
    }

    @Override
    public void save() throws ParseException {
        Settings.cinemaModeZoomMultiplier= sliderZoomMul.getValue();
        Settings.cinemaModeSwitchTime = parseInt(textFieldTime.getText(), "Cinema mode switch time", 1, 3600);
        Settings.cinemaModeOnStartup = chkBoxEnableOnStartup.isSelected();
        Settings.cinemaModeReenable = chkBoxReenable.isSelected();
    }

    @Override
    public String getTitle() {
        return "Cinema Mode";
    }
}
