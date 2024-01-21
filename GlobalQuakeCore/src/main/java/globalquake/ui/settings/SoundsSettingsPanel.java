package globalquake.ui.settings;

import globalquake.core.Settings;
import globalquake.sounds.GQSound;
import globalquake.sounds.Sounds;

import javax.swing.*;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;

public class SoundsSettingsPanel extends SettingsPanel {

    private JSlider sliderMasterVolume;

    public SoundsSettingsPanel() {
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        add(createMasterVolumeSlider());
        add(createIndividualSoundsPanel());
    }

    private Component createIndividualSoundsPanel() {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        for(GQSound gqSound : Sounds.ALL_ACTUAL_SOUNDS){
            var borderLayout = new BorderLayout();
            borderLayout.setHgap(10);
            borderLayout.setVgap(4);
            JPanel soundPanel = new JPanel(borderLayout);
            //soundPanel.setLayout(new BoxLayout(soundPanel, BoxLayout.X_AXIS));
            soundPanel.setBorder(BorderFactory.createRaisedBevelBorder());
            soundPanel.setSize(300,300);

            JLabel label = new JLabel(gqSound.getFilename());
            label.setPreferredSize(new Dimension(160, 40));

            soundPanel.add(label, BorderLayout.WEST);

            JPanel volumePanel = new JPanel(new BorderLayout());
            volumePanel.add(new JLabel("Volume"), BorderLayout.NORTH);

            JSlider volumeSlider = new JSlider(SwingConstants.HORIZONTAL,0,100, (int) (gqSound.volume * 100.0));
            volumeSlider.setMajorTickSpacing(10);
            volumeSlider.setMinorTickSpacing(2);
            volumeSlider.setPaintTicks(true);
            volumeSlider.setPaintLabels(true);

            volumeSlider.addChangeListener(changeEvent -> gqSound.volume = volumeSlider.getValue() / 100.0);

            volumePanel.add(volumeSlider, BorderLayout.CENTER);

            soundPanel.add(volumePanel);

            JButton testSoundButton = new JButton("Test");
            testSoundButton.addActionListener(new AbstractAction() {
                @Override
                public void actionPerformed(ActionEvent actionEvent) {
                    Sounds.playSound(gqSound);
                }
            });

            JPanel p = new JPanel();
            p.add(testSoundButton);
            soundPanel.add(p, BorderLayout.EAST);

            panel.add(soundPanel);
        }

        return new JScrollPane(panel);
    }

    private Component createMasterVolumeSlider(){
        sliderMasterVolume = HypocenterAnalysisSettingsPanel.createSettingsSlider(0, 100, 10, 2);

        JLabel label = new JLabel();
        ChangeListener changeListener = changeEvent -> {
            label.setText("Master Volume: %d%%".formatted(
                    sliderMasterVolume.getValue()));
            Settings.globalVolume = sliderMasterVolume.getValue();
        };

        sliderMasterVolume.addChangeListener(changeListener);

        sliderMasterVolume.setValue(Settings.globalVolume);
        changeListener.stateChanged(null);

        return HypocenterAnalysisSettingsPanel.createCoolLayout(sliderMasterVolume, label, null,
                null);
    }

    @Override
    public void save() throws NumberFormatException {
        // TODO save volumes
    }

    @Override
    public String getTitle() {
        return "Sounds";
    }
}
